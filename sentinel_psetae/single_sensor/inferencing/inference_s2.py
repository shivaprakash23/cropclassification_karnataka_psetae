"""
Inference script for Sentinel-2 PSETAE model, adapted from training code
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

class S2InferenceDataset(data.Dataset):
    def __init__(self, folder, npixel=128, minimum_sampling=None):
        super(S2InferenceDataset, self).__init__()
        self.data_folder = os.path.join(folder, 'DATA')
        self.meta_folder = os.path.join(folder, 'META')
        self.npixel = npixel
        self.minimum_sampling = minimum_sampling

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
        """Get a single sample.
        
        Returns:
            tuple: ((x, mask), label, dates)
                x: tensor of shape [T, C, S] where T is time steps, C is channels (10 for S2), S is pixels
                mask: tensor of shape [S] indicating valid pixels
                label: class label (0-based)
                dates: tensor of shape [T] with temporal positions
        """
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
        
        # Handle temporal sampling for S2 data
        T = x.shape[0]
        if T > self.minimum_sampling:
            # Randomly sample minimum_sampling time points
            t_idx = sorted(random.sample(range(T), self.minimum_sampling))
            x = x[t_idx]
            if self.dates:
                # Use sequence positions instead of actual dates for now
                dates = torch.tensor(t_idx, dtype=torch.float32)
            else:
                dates = torch.tensor(t_idx, dtype=torch.float32)
        else:
            # Pad with zeros if less than minimum_sampling
            padded_x = torch.zeros((self.minimum_sampling, x.shape[1], x.shape[2]), dtype=torch.float32)
            padded_x[:T] = x
            x = padded_x
            if self.dates:
                # Use sequence positions instead of actual dates for now
                dates = torch.tensor(list(range(T)) + [0] * (self.minimum_sampling - T), dtype=torch.float32)
            else:
                dates = torch.tensor(list(range(T)) + [0] * (self.minimum_sampling - T), dtype=torch.float32)
            
        # Get label if available
        if self.labels and str(sample_id) in self.labels:
            label = self.labels[str(sample_id)]
        else:
            label = -1
            
        return (x, mask), torch.tensor(label), dates

def custom_collate(batch):
    """Custom collate function for S2 data.
    Expected model input format:
    - x: [B, T, C, S] where B is batch size, T is time steps, C is channels, S is pixels
    - mask: [B, T, S] where B is batch size, T is time steps, S is pixels
    - dates: [B, T] where B is batch size, T is time steps
    """
    xs = []
    masks = []
    batch_y = []
    batch_dates = []
    
    for (x, mask), y, dates in batch:
        # x shape: [T, C, S] -> [T, C, S]
        xs.append(x.float())
        # Expand mask to match temporal dimension
        # mask shape: [S] -> [T, S]
        T = x.shape[0]
        mask_expanded = mask.unsqueeze(0).expand(T, -1)
        masks.append(mask_expanded.float())
        batch_y.append(y)
        batch_dates.append(dates)
    
    # Stack all tensors
    # x: [B, T, C, S]
    x = torch.stack(xs)
    # mask: [B, T, S]
    mask = torch.stack(masks)
    y = torch.stack(batch_y)
    # dates: [B, T]
    dates = torch.stack(batch_dates)
    
    return (x, mask), y, dates

def load_model(model_path, device):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with same configuration as training
    model = PseTae(
        input_dim=10,  # S2 has 10 bands
        mlp1=[10, 32, 64],
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
        len_max_seq=210,  # Match the model's sequence length
        positions=None,
        mlp4=[128, 64, 32, 12]  # 12 output classes
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model

def main():
    parser = ArgumentParser(description='Inference script for Sentinel-2 PSETAE model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint (.pth.tar file)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to directory containing DATA and META folders')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    parser.add_argument('--minimum_sampling', type=int, default=24,
                      help='Minimum number of temporal samples to use (if None, uses all available samples)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    output_dir = os.path.join(args.output_dir, 'results')
    
    # Create dataset and dataloader
    dataset = S2InferenceDataset(args.data_dir, minimum_sampling=args.minimum_sampling)
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                                shuffle=False, collate_fn=custom_collate)
    
    # Load model
    model = load_model(args.model_path, args.device)
    model.eval()
    
    # Run inference
    all_preds = []
    all_labels = []
    all_probs = []
    sample_ids = []
    
    print(f"\nProcessing {len(dataset)} samples...")
    
    for batch_idx, ((x, mask), y, dates) in enumerate(tqdm(data_loader)):
        # Get batch sample IDs
        start_idx = batch_idx * args.batch_size
        end_idx = start_idx + y.shape[0]
        batch_ids = dataset.sample_ids[start_idx:end_idx]
        sample_ids.extend(batch_ids)
        
        # Move to device
        x = x.to(args.device)
        mask = mask.to(args.device)
        y = y.to(args.device)
        
        # Get predictions
        with torch.no_grad():
            # Debug prints
            print(f"Input shapes - x: {x.shape}, mask: {mask.shape}, dates: {dates.shape}")
            prediction = model((x, mask), dates)
            probabilities = F.softmax(prediction, dim=1)
        
        # Store predictions and true labels
        all_preds.extend(prediction.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probabilities.cpu().numpy())
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'parcel_id': sample_ids,
        'true_class': all_labels,
        'predicted_class': all_preds,
        'confidence': [probs[pred] for probs, pred in zip(all_probs, all_preds)]
    })
    
    # Add probability columns for each class
    for i in range(10):
        results_df[f'prob_class_{i}'] = all_probs[:, i]
    
    # Save predictions to CSV
    results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Generate confusion matrix
    plt.figure(figsize=(15, 10))
    class_labels = [str(i) for i in range(10)]  # Classes 0-9
    
    # Create heatmap with raw counts (no normalization)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    sns.heatmap(conf_matrix,
              annot=True,
              fmt='d',
              cmap='OrRd',  # Using OrRd colormap for darker reds
              xticklabels=class_labels,
              yticklabels=class_labels,
              linewidths=0.5)
    
    # Customize appearance
    plt.title('Confusion Matrix', pad=20, size=12)
    plt.xlabel('Predicted Label (0-9)', size=10)
    plt.ylabel('True Label (0-9)', size=10)
    
    # Adjust label parameters
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Calculate class-wise metrics
    def calculate_class_metrics(df, conf_threshold=None):
        if conf_threshold is not None:
            df = df[df['confidence'] >= conf_threshold].copy()
        
        if len(df) == 0:
            return None
        
        metrics_dict = {}
        for class_idx in range(10):
            class_mask = df['true_class'] == class_idx
            if not class_mask.any():
                continue
            
            true_pos = ((df['true_class'] == class_idx) & (df['predicted_class'] == class_idx)).sum()
            false_pos = ((df['true_class'] != class_idx) & (df['predicted_class'] == class_idx)).sum()
            false_neg = ((df['true_class'] == class_idx) & (df['predicted_class'] != class_idx)).sum()
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            support = class_mask.sum()
            avg_conf = df[df['true_class'] == class_idx]['confidence'].mean() if class_mask.any() else 0
            
            metrics_dict[class_idx] = {
                'f1-score': f1,
                'precision': precision,
                'recall': recall,
                'support': support,
                'avg_conf': avg_conf
            }
        
        return metrics_dict
    
    # Calculate metrics for different confidence thresholds
    all_metrics = calculate_class_metrics(results_df)
    high_conf_metrics = calculate_class_metrics(results_df, 0.80)
    med_conf_metrics = calculate_class_metrics(results_df[(results_df['confidence'] >= 0.50) & (results_df['confidence'] < 0.80)])
    low_conf_metrics = calculate_class_metrics(results_df[results_df['confidence'] < 0.50])
    
    # Create performance summary file
    with open(os.path.join(output_dir, 'model_performance_summary.txt'), 'w') as f:
        # Overall accuracy
        accuracy = (results_df['true_class'] == results_df['predicted_class']).mean()
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        # Class-wise metrics sorted by F1-score
        f.write("Class-wise Metrics (sorted by F1-score):\n")
        f.write("Class   F1-score  Precision  Recall  Avg.Conf  Support\n")
        f.write("-" * 55 + "\n")
        
        sorted_classes = sorted(all_metrics.keys(), key=lambda x: all_metrics[x]['f1-score'], reverse=True)
        for class_idx in sorted_classes:
            metrics = all_metrics[class_idx]
            f.write(f"{class_idx:^5d}   {metrics['f1-score']:.3f}    {metrics['precision']:.3f}     {metrics['recall']:.3f}   {metrics['avg_conf']:.3f}    {metrics['support']}\n")
        
        f.write("\nResults saved to " + output_dir + "\n\n")
        
        # High confidence performance
        if high_conf_metrics:
            f.write("High Confidence (>=0.80) Performance by Class:\n\n")
            f.write("Class   F1-score  Precision  Recall  Support\n")
            f.write("-" * 45 + "\n")
            for class_idx in sorted_classes:
                if class_idx in high_conf_metrics:
                    m = high_conf_metrics[class_idx]
                    f.write(f"{class_idx:^5d}   {m['f1-score']:.3f}    {m['precision']:.3f}     {m['recall']:.3f}    {m['support']}\n")
        
        # Medium confidence performance
        if med_conf_metrics:
            f.write("\nMedium Confidence (0.50-0.80) Performance by Class:\n\n")
            f.write("Class   F1-score  Precision  Recall  Support\n")
            f.write("-" * 45 + "\n")
            for class_idx in sorted_classes:
                if class_idx in med_conf_metrics:
                    m = med_conf_metrics[class_idx]
                    f.write(f"{class_idx:^5d}   {m['f1-score']:.3f}    {m['precision']:.3f}     {m['recall']:.3f}    {m['support']}\n")
        
        # Low confidence performance
        if low_conf_metrics:
            f.write("\nLow Confidence (<0.50) Performance by Class:\n\n")
            f.write("Class   F1-score  Precision  Recall  Support\n")
            f.write("-" * 45 + "\n")
            for class_idx in sorted_classes:
                if class_idx in low_conf_metrics:
                    m = low_conf_metrics[class_idx]
                    f.write(f"{class_idx:^5d}   {m['f1-score']:.3f}    {m['precision']:.3f}     {m['recall']:.3f}    {m['support']}\n")
        
        # Additional statistics
        total_samples = len(results_df)
        high_conf = (results_df['confidence'] >= 0.80).sum()
        med_conf = ((results_df['confidence'] >= 0.50) & (results_df['confidence'] < 0.80)).sum()
        low_conf = (results_df['confidence'] < 0.50).sum()
        
        f.write(f"\nTotal samples processed: {total_samples}\n")
        f.write("\nConfidence distribution:\n")
        f.write(f"High confidence (>=0.80): {high_conf} samples ({high_conf/total_samples*100:.2f}%)\n")
        f.write(f"Medium confidence (0.50-0.80): {med_conf} samples ({med_conf/total_samples*100:.2f}%)\n")
        f.write(f"Low confidence (<0.50): {low_conf} samples ({low_conf/total_samples*100:.2f}%)\n")
        
        f.write("\nClass distribution:\n")
        class_counts = pd.Series(0, index=range(10))
        counts = results_df['predicted_class'].value_counts()
        class_counts.update(counts)
        for class_idx in range(10):
            f.write(f"Class {class_idx}: {int(class_counts[class_idx])} samples\n")
        
        f.write(f"\nMean confidence: {results_df['confidence'].mean():.4f}\n")
    
    # Print summary to console
    print(f"\nResults saved to {output_dir}")
    with open(os.path.join(output_dir, 'model_performance_summary.txt'), 'r') as f:
        print(f.read())

if __name__ == '__main__':
    main()
