"""
Adapted testing script for HLS L30 PSETAE model using the same approach as training evaluation
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchnet as tnt
from tqdm import tqdm
from argparse import ArgumentParser
import sys
sys.path.append('D:/Semester4/ProjectVijayapur/psetae/psetae_all5models/3_hls_l30_single_sensor')
from models.stclassifier import PseTae
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class GEEDataset(Dataset):
    """Dataset for GEE-extracted data using same format as training."""
    def __init__(self, data_dir):
        self.data_path = os.path.join(data_dir, 'DATA')
        self.meta_path = os.path.join(data_dir, 'META')
        
        # Get all sample IDs
        self.sample_ids = [f.split('.')[0] for f in os.listdir(self.data_path) if f.endswith('.npy')]
        
        # Load labels (keep as 1 to n)
        with open(os.path.join(self.meta_path, 'labels.json'), 'r') as f:
            labels_data = json.load(f)['CODE_GROUP']
            self.labels = {pid: int(label) for pid, label in labels_data.items()}
        
        # Load dates
        with open(os.path.join(self.meta_path, "dates.json"), "r") as f:
            self.dates_dict = json.load(f)
    
    def __len__(self):
        return len(self.sample_ids)
    
    def get_batch_ids(self, indices):
        """Get parcel IDs for the given indices."""
        return [self.sample_ids[i] for i in indices]
    
    def __getitem__(self, idx):
        self.current_idx = idx  # Store current index for get_batch_ids
        sample_id = self.sample_ids[idx]
        
        # Load data array
        data_file = os.path.join(self.data_path, f"{sample_id}.npy")
        data_array = np.load(data_file)  # Shape: [T, C, S]
        
        # Get dates for this sample
        dates = self.dates_dict[str(sample_id)]
        
        # Get label (0 to n-1)
        label = self.labels.get(str(sample_id), 1)  # Default to class 1 if not found
        label = label - 1  # Convert to 0-based
        
        # Convert to torch tensors
        x = torch.from_numpy(data_array).float()  # Shape: [T, C, S]
        y = torch.tensor(label).long()
        
        # Create mask for valid pixels (all pixels are valid)
        mask = torch.ones(x.shape[-1], dtype=torch.bool)
        
        return (x, mask), y, dates

def collate_fn(batch):
    """Custom collate function to handle variable-sized inputs."""
    # Each item in batch is ((x, mask), y, dates)
    x_tensors = []
    mask_tensors = []
    y_tensors = []
    dates_list = []
    
    # Find the maximum dimensions in this batch
    max_pixels = max(x.shape[-1] for (x, mask), _, _ in batch)
    max_time = max(x.shape[0] for (x, mask), _, _ in batch)
    
    for (x, mask), y, dates in batch:
        # Pad temporal dimension if needed
        if x.shape[0] < max_time:
            time_pad = max_time - x.shape[0]
            x = F.pad(x, (0, 0, 0, 0, 0, time_pad))  # Pad time dimension
        
        # Pad pixel dimension if needed
        if x.shape[-1] < max_pixels:
            pixel_pad = max_pixels - x.shape[-1]
            x = F.pad(x, (0, pixel_pad))  # Pad pixel dimension
            mask = F.pad(mask, (0, pixel_pad))  # Pad mask too
        
        x_tensors.append(x)
        mask_tensors.append(mask)
        y_tensors.append(y)
        dates_list.append(dates)
    
    # Stack tensors
    x_batch = torch.stack(x_tensors)  # Shape: [B, T, C, S]
    mask_batch = torch.stack(mask_tensors)  # Shape: [B, S]
    y_batch = torch.stack(y_tensors)  # Shape: [B]
    
    return (x_batch, mask_batch), y_batch, dates_list

def recursive_todevice(x, device):
    """Recursively moves nested tensors to device."""
    if isinstance(x, (list, tuple)):
        return [recursive_todevice(x_i, device) for x_i in x]
    else:
        return x.to(device)

def load_model(model_path, device):
    """Load model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with same configuration as training
    model = PseTae(
        input_dim=8,
        mlp1=[8, 32, 64],
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
        positions=None,
        mlp4=[128, 64, 32, 10]  # 10 output classes
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, data_loader, device):
    """Evaluate model using same approach as training evaluation."""
    model.eval()
    
    y_true = []
    y_pred = []
    probs = []
    parcel_ids = []
    
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    
    for batch_idx, ((x, mask), y, dates) in enumerate(tqdm(data_loader, desc="Evaluating")):
        # Get parcel IDs for this batch
        start_idx = batch_idx * data_loader.batch_size
        end_idx = start_idx + y.shape[0]
        batch_ids = data_loader.dataset.get_batch_ids(range(start_idx, end_idx))
        parcel_ids.extend(batch_ids)
        
        # Store true labels (0-based)
        batch_true_labels = list(map(int, y))
        y_true.extend(batch_true_labels)
        
        # Move to device
        x = x.to(device)  # Shape: [B, T, C, S]
        mask = mask.to(device)  # Shape: [B, S]
        y = y.to(device)  # Shape: [B]
        
        with torch.no_grad():
            # Get predictions
            prediction = model((x, mask), dates)
            
            # Get predictions for all classes
            prediction = prediction
            
            # Update accuracy meter
            acc_meter.add(prediction.detach(), y)
            
            # Get predictions and probabilities
            softmax_probs = F.softmax(prediction, dim=1).detach().cpu().numpy()
            batch_preds = prediction.argmax(dim=1).cpu().numpy()  # Keep as 0-based
            
            # Store predictions and probabilities
            probs.extend(softmax_probs)
            y_pred.extend(batch_preds)
    
    # Calculate metrics
    accuracy = acc_meter.value()[0]
    
    # Ensure labels are in correct order (0-9)
    label_order = list(range(10))
    conf_mat = confusion_matrix(y_true, y_pred, labels=label_order)
    class_report = classification_report(y_true, y_pred, labels=label_order, zero_division=0)
    
    print("\nClassification Report:")
    print(class_report)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_mat,
        'classification_report': class_report,
        'y_true': y_true,
        'y_pred': y_pred,
        'probabilities': probs,
        'parcel_ids': parcel_ids
    }

def main():
    parser = ArgumentParser(description='Adapted testing script for HLS L30 PSETAE model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint (.pth.tar file)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to directory containing DATA and META folders')
    parser.add_argument('--output_dir', type=str, default='.',
                      help='Directory to save results')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    output_dir = os.path.join(args.output_dir, 'results')
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Create dataset and dataloader
    dataset = GEEDataset(args.data_dir)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Evaluate model
    metrics = evaluate_model(model, data_loader, args.device)
    
    # Convert numpy types to Python types for JSON serialization
    metrics_json = {
        'accuracy': float(metrics['accuracy']),
        'classification_report': metrics['classification_report'],
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'y_true': [int(x) for x in metrics['y_true']],
        'y_pred': [int(x) for x in metrics['y_pred']],
        'probabilities': [p.tolist() for p in metrics['probabilities']]
    }
    
    # Save metrics to JSON
    output_file = os.path.join(output_dir, 'metrics.json')
    with open(output_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Create DataFrame for CSV output
    data = {
        'parcel_id': metrics['parcel_ids'],
        'true_class': metrics['y_true'],  # Keep true labels 0-based
        'predicted_class': metrics['y_pred'],  # Keep predictions 0-based
        'confidence': None,  # Will be filled below
    }
    
    # Add confidence scores (probability of predicted class)
    data['confidence'] = [float(probs[pred]) for probs, pred in zip(metrics['probabilities'], metrics['y_pred'])]
    
    # Add probability columns for each class
    for i in range(10):  # 10 classes
        data[f'prob_class_{i}'] = [float(p[i]) for p in metrics['probabilities']]
    
    # Create DataFrame
    results_df = pd.DataFrame(data)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, 'predictions.csv')
    results_df.to_csv(csv_file, index=False)
    
    # Generate confusion matrix with style matching the first image
    plt.figure(figsize=(15, 10))
    class_labels = [str(i) for i in range(10)]  # Classes 0-9
    
    # Create heatmap with raw counts (no normalization)
    sns.heatmap(metrics['confusion_matrix'],
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
