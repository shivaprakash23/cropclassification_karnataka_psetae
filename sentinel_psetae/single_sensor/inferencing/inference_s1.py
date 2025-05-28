"""
Inference script for Sentinel-1 PSETAE model, adapted from training code
"""

import torch
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import os
import json
from models.stclassifier import PseTae
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class S1InferenceDataset(data.Dataset):
    def __init__(self, folder, npixel=64):
        super(S1InferenceDataset, self).__init__()
        self.data_folder = os.path.join(folder, 'DATA')
        self.meta_folder = os.path.join(folder, 'META')
        self.npixel = npixel

        # Get sample IDs
        l = [f for f in os.listdir(self.data_folder) if f.endswith('.npy')]
        self.sample_ids = [f.split('.')[0] for f in l]
        
        # Load dates if available
        try:
            with open(os.path.join(self.meta_folder, 'dates.json'), 'r') as f:
                dates_dict = json.load(f)
                self.dates = {str(k): v for k, v in dates_dict.items()}
        except:
            self.dates = None
            
        # Load labels if available
        try:
            with open(os.path.join(self.meta_folder, 'labels.json'), 'r') as f:
                labels_dict = json.load(f)['CODE_GROUP']
                self.labels = {str(k): int(v)-1 for k, v in labels_dict.items()}  # Convert to 0-based
        except:
            self.labels = None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load data array
        data_file = os.path.join(self.data_folder, f"{sample_id}.npy")
        data_array = np.load(data_file)  # Shape: [T, C, S]
        
        # Convert to tensor
        x = torch.from_numpy(data_array).float()
        
        # Sample or pad pixels
        n_available_pixels = x.shape[-1]
        if n_available_pixels > self.npixel:
            # Sample random pixels
            pixel_indices = sorted(random.sample(range(n_available_pixels), self.npixel))
            x = x[:, :, pixel_indices]
            mask = torch.ones(self.npixel, dtype=torch.float32)
        else:
            # Pad with zeros
            padded_x = torch.zeros((x.shape[0], x.shape[1], self.npixel), dtype=torch.float32)
            padded_x[:, :, :n_available_pixels] = x
            x = padded_x
            
            # Create mask (1 for real pixels, 0 for padded)
            mask = torch.zeros(self.npixel, dtype=torch.float32)
            mask[:n_available_pixels] = 1.0
            
        # Get dates (use sequence position if dates not available)
        if self.dates and str(sample_id) in self.dates:
            dates = torch.tensor(list(range(len(self.dates[str(sample_id)]))), dtype=torch.float32)
        else:
            dates = torch.tensor(list(range(x.shape[0])), dtype=torch.float32)
            
        # Get label if available
        if self.labels and str(sample_id) in self.labels:
            label = self.labels[str(sample_id)]
        else:
            label = -1
            
        return (x, mask), torch.tensor(label), dates

def custom_collate(batch):
    """Custom collate function for S1 data."""
    xs = []
    masks = []
    batch_y = []
    batch_dates = []
    
    for (x, mask), y, dates in batch:
        xs.append(x.float())
        masks.append(mask.float())
        batch_y.append(y)
        batch_dates.append(dates)
    
    # Get max sequence length
    max_len = max(x.shape[0] for x in xs)
    
    # Pad sequences
    padded_x = []
    padded_dates = []
    
    for x, dates in zip(xs, batch_dates):
        curr_len = x.shape[0]
        if curr_len < max_len:
            padding = (0, 0, 0, 0, 0, max_len - curr_len)
            x_pad = F.pad(x, padding)
            date_padding = dates[-1].repeat(max_len - curr_len)
            dates = torch.cat([dates, date_padding])
        else:
            x_pad = x
        padded_x.append(x_pad.float())
        padded_dates.append(dates.float())
    
    # Stack tensors
    x_stack = torch.stack(padded_x)
    expanded_masks = [m.unsqueeze(0).expand(max_len, -1).float() for m in masks]
    mask_stack = torch.stack(expanded_masks)
    y_stack = torch.stack([torch.tensor(y, dtype=torch.int64) for y in batch_y])
    dates_stack = torch.stack(padded_dates)
    
    return (x_stack, mask_stack), y_stack, dates_stack

def load_model(model_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = PseTae(
        input_dim=2,  # S1 bands: VV, VH
        mlp1=[2, 32, 64],
        pooling='mean_std',
        mlp2=[128, 128],
        with_extra=False,
        extra_size=0,
        n_head=4,
        d_k=32,
        d_model=None,
        mlp3=[512, 128, 128],
        dropout=0.0,
        T=366,
        len_max_seq=210,
        positions='order',
        mlp4=[128, 64, 32, 12]
    )
    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model

def main():
    parser = ArgumentParser(description='Inference script for S1 PSETAE model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint (.pth.tar file)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to directory containing DATA folder')
    parser.add_argument('--output_dir', type=str, default='.',
                      help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create dataset and dataloader
    dataset = S1InferenceDataset(args.data_dir, npixel=64)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Process all samples
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_sample_ids = []
    
    print(f"\nProcessing {len(dataset)} samples...")
    with torch.no_grad():
        for (x, mask), y, dates in tqdm(dataloader):
            # Move to device
            x = x.to(args.device)
            mask = mask.to(args.device)
            dates = dates.to(args.device)
            
            # Get predictions
            output = model((x, mask), dates)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())
            all_sample_ids.extend(dataset.sample_ids[i] for i in range(len(y)))
    
    # Create results DataFrame
    results = []
    for i, sample_id in enumerate(all_sample_ids):
        results.append({
            'sample_id': sample_id,
            'true_class': all_true_labels[i],
            'predicted_class': all_predictions[i],
            'confidence': all_probabilities[i][all_predictions[i]],
            **{f'prob_class_{j}': prob for j, prob in enumerate(all_probabilities[i])}
        })
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(args.output_dir, 'predictions.csv')
    results_df.to_csv(csv_file, index=False)
    
    # Print accuracy and generate metrics if true labels are available
    if dataset.labels is not None:
        # Calculate and print overall accuracy
        accuracy = (results_df['true_class'] == results_df['predicted_class']).mean()
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Generate confusion matrix for classes 0-9
        plt.figure(figsize=(12, 10))
        class_labels = [str(i) for i in range(10)]  # Classes 0-9
        # Filter predictions to only include classes 0-9
        mask_0_9 = (results_df['true_class'] < 10) & (results_df['predicted_class'] < 10)
        conf_matrix = confusion_matrix(
            results_df[mask_0_9]['true_class'], 
            results_df[mask_0_9]['predicted_class'],
            labels=list(range(10))  # Explicitly specify labels 0-9
        )
        img = sns.heatmap(conf_matrix,
                         annot=True,
                         fmt='d',
                         linewidths=0.5,
                         cmap='OrRd',
                         xticklabels=class_labels,
                         yticklabels=class_labels)
        img.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
        img.set(ylabel="True Label (0-9)", xlabel="Predicted Label (0-9)", title="Confusion Matrix")
        img.figure.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), bbox_inches='tight')
        img.get_figure().clf()
        
        # Generate metrics for high-confidence predictions (classes 0-9)
        high_conf_mask = (results_df['confidence'] >= 0.80) & (results_df['true_class'] < 10) & (results_df['predicted_class'] < 10)
        if high_conf_mask.any():
            high_conf_df = results_df[high_conf_mask]
            print("\nClassification Report (High Confidence Predictions >= 0.80, Classes 0-9):")
            print(classification_report(high_conf_df['true_class'], high_conf_df['predicted_class'], labels=list(range(10))))
        
        # Print summary statistics
        print("\nPrediction Summary:")
        print(f"Total samples processed: {len(results_df)}\n")
        
        # Calculate confidence distribution
        high_conf = (results_df['confidence'] >= 0.80).sum()
        med_conf = ((results_df['confidence'] >= 0.50) & (results_df['confidence'] < 0.80)).sum()
        low_conf = (results_df['confidence'] < 0.50).sum()
        
        print("Confidence distribution:")
        print(f"High confidence (>=0.80): {high_conf} samples ({high_conf/len(results_df)*100:.2f}%)")
        print(f"Medium confidence (0.50-0.80): {med_conf} samples ({med_conf/len(results_df)*100:.2f}%)")
        print(f"Low confidence (<0.50): {low_conf} samples ({low_conf/len(results_df)*100:.2f}%)\n")
        
        print("Class distribution (0-9):")
        class_counts = pd.Series(0, index=range(10))  # Initialize counts for classes 0-9
        counts = results_df[results_df['predicted_class'] < 10]['predicted_class'].value_counts()
        class_counts.update(counts)  # Update with actual counts
        for class_idx in range(10):
            print(f"Class {class_idx}: {int(class_counts[class_idx])} samples")
        mean_conf = results_df['confidence'].mean()
        print(f"\nMean confidence: {mean_conf:.4f}")
        
        # Save class-wise F1 scores
        class_metrics = pd.DataFrame(classification_report(results_df['true_class'], results_df['predicted_class'], output_dict=True)).transpose()
        class_metrics.to_csv(os.path.join(args.output_dir, 'class_metrics.csv'))
        
        # Save performance summary to text file
        summary_text = f"Overall Accuracy: {accuracy:.4f}\n\n"
        summary_text += "Classification Report (High Confidence Predictions >= 0.80, Classes 0-9):\n"
        summary_text += classification_report(high_conf_df['true_class'], high_conf_df['predicted_class'], labels=list(range(10)))
        summary_text += "\nPrediction Summary:\n"
        summary_text += f"Total samples processed: {len(results_df)}\n\n"
        summary_text += "Confidence distribution:\n"
        summary_text += f"High confidence (>=0.80): {high_conf} samples ({high_conf/len(results_df)*100:.2f}%)\n"
        summary_text += f"Medium confidence (0.50-0.80): {med_conf} samples ({med_conf/len(results_df)*100:.2f}%)\n"
        summary_text += f"Low confidence (<0.50): {low_conf} samples ({low_conf/len(results_df)*100:.2f}%)\n\n"
        summary_text += "Class distribution (0-9):\n"
        for class_idx in range(10):
            summary_text += f"Class {class_idx}: {int(class_counts[class_idx])} samples\n"
        summary_text += f"\nMean confidence: {mean_conf:.4f}"
        
        # Write summary to file
        with open(os.path.join(args.output_dir, 'model_performance_summary.txt'), 'w') as f:
            f.write(summary_text)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main()
