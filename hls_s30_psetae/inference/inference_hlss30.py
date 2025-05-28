"""
Inference script for HLS S30 PSETAE model using GEE-extracted data
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
import random
import torch.nn.functional as F
from models.stclassifier import PseTae
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

class HLSS30Inference:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize inference for HLS L30 PSETAE model.
        
        Args:
            model_path (str): Path to the model checkpoint (.pth.tar file)
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Use hardcoded config from training command
        self.config = {
            'input_dim': 10,  # S30 bands: B2-B8, B8A, B11-B12
            'mlp1': [10, 32, 64],
            'mlp2': [128, 128],
            'num_classes': 10,  # 10 classes (0-9)
            'geomfeat': 0,
            'confidence_threshold': 0.80  # Threshold for high confidence predictions
        }
        
        # Initialize model
        self.model = PseTae(
            input_dim=10,  # S30 bands: B2-B8, B8A, B11-B12
            mlp1=[10, 32, 64],  # First layer matches input dimension
            pooling='mean_std',
            mlp2=[128, 128],  # From training command --mlp2 [128,128]
            with_extra=False,  # No geometric features (--geomfeat 0)
            extra_size=0,  # No geometric features
            n_head=4,  # From training command --n_head 4
            d_k=32,  # From training command --d_k 32
            d_model=None,  # No input projection
            mlp3=[512, 128, 128],  # From training command --mlp3 [512,128,128]
            dropout=0.0,  # Set to 0 for inference
            T=366,  # From training command --T 366
            len_max_seq=210,  # From training command --lms 210
            positions=None,  # Using None for default behavior
            mlp4=[128, 64, 32, 10]  # From training command --mlp4 [128,64,32,10]
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Load normalization parameters if available
        self.norm_params = self.config.get('norm_params', None)
    
    def normalize_input(self, x):
        """Normalize input data if normalization parameters are available."""
        if self.norm_params is not None:
            mean = torch.tensor(self.norm_params[0], device=self.device)
            std = torch.tensor(self.norm_params[1], device=self.device)
            x = (x - mean) / std
        return x
    
    def process_dates(self, dates):
        """Convert date strings to temporal positions."""
        date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        ref_date = date_objects[0]
        days = [(d - ref_date).days for d in date_objects]
        return torch.tensor(days, dtype=torch.float32)
    
    def prepare_single_sample(self, data_array, dates, n_pixels=16, minimum_sampling=8):
        """
        Prepare a single sample for inference.
        
        Args:
            data_array (numpy.ndarray): Shape (T, C, S) where:
                T: number of timestamps
                C: number of channels
                S: number of pixels
            dates (list): List of date strings in format 'YYYY-MM-DD'
            n_pixels (int): Number of pixels to sample (default: 16)
            minimum_sampling (int): Number of timesteps to sample (default: 8)
            
        Returns:
            tuple: Processed tensors ready for model input
        """
        # Convert to tensor and handle pixels first
        n_available_pixels = data_array.shape[2]
        if n_available_pixels > n_pixels:
            # Randomly sample pixels
            pixel_indices = sorted(random.sample(range(n_available_pixels), n_pixels))
            data_array = data_array[:, :, pixel_indices]
        elif n_available_pixels < n_pixels:
            # Pad by repeating pixels
            repeats = n_pixels // n_available_pixels + 1
            data_array = np.repeat(data_array, repeats, axis=2)[:, :, :n_pixels]
        
        # Now handle timesteps
        n_timesteps = data_array.shape[0]
        if n_timesteps > minimum_sampling:
            # Randomly sample timesteps
            time_indices = sorted(random.sample(range(n_timesteps), minimum_sampling))
            data_array = data_array[time_indices]
            dates = [dates[i] for i in time_indices]
        elif n_timesteps < minimum_sampling:
            # Pad with zeros
            pad_size = minimum_sampling - n_timesteps
            padding = np.zeros((pad_size, data_array.shape[1], data_array.shape[2]))
            data_array = np.concatenate([data_array, padding], axis=0)
            # Pad dates with the last date
            last_date = dates[-1] if dates else '2024-01-01'
            dates.extend([last_date] * pad_size)
        
        # Convert to tensor
        x = torch.from_numpy(data_array).float()
        
        # Create pixel mask (1 for real pixels, 0 for padded)
        mask = torch.ones(n_pixels, dtype=torch.float32)
        if n_available_pixels < n_pixels:
            mask[n_available_pixels:] = 0
        
        # Process dates
        date_positions = self.process_dates(dates)
        
        # Add batch dimension and normalize
        x = x.unsqueeze(0)  # (1, T, C, S)
        mask = mask.unsqueeze(0)  # (1, S)
        date_positions = date_positions.unsqueeze(0)  # (1, T)
        
        # Normalize input
        x = self.normalize_input(x)
        
        return (x, mask), date_positions
    
    @torch.no_grad()
    def predict(self, data_array, dates):
        """
        Run inference on a single sample.
        
        Args:
            data_array (numpy.ndarray): Shape (T, C, S)
            dates (list): List of date strings in format 'YYYY-MM-DD'
            
        Returns:
            dict: Dictionary containing:
                - predicted_class: The predicted class label
                - probabilities: Softmax probabilities for all classes
                - confidence: Confidence score for the prediction
        """
        # Prepare input
        (x, mask), dates = self.prepare_single_sample(data_array, dates)
        
        # Move to device
        x = x.to(self.device)
        mask = mask.to(self.device)
        dates = dates.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model((x, mask), dates)
            probabilities = F.softmax(output, dim=1)
        
        # Get prediction and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'probabilities': probabilities[0].cpu().numpy(),
            'confidence': confidence
        }

def load_gee_sample(data_path, meta_path, sample_id):
    """
    Load a sample from GEE-extracted data.
    
    Args:
        data_path (str): Path to the DATA directory containing .npy files
        meta_path (str): Path to the META directory containing json files
        sample_id (str): ID of the sample to load
        
    Returns:
        tuple: (data_array, dates)
            - data_array: numpy array of shape (T, C, S)
            - dates: list of date strings
    """
    # Load the data array
    data_file = os.path.join(data_path, f"{sample_id}.npy")
    data_array = np.load(data_file)
    
    # Load dates
    with open(os.path.join(meta_path, "dates.json"), "r") as f:
        dates_dict = json.load(f)
        dates = dates_dict[str(sample_id)]
    
    return data_array, dates

if __name__ == '__main__':
    parser = ArgumentParser(description='Run inference with HLS S30 PSETAE model on GEE-extracted data')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint (.pth.tar file)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to directory containing DATA and META folders')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                      help='Path to save predictions CSV')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_path = os.path.join(args.data_dir, 'DATA')
    meta_path = os.path.join(args.data_dir, 'META')
    
    # Initialize inference
    inferencer = HLSS30Inference(args.model_path, device=args.device)
    
    # Get list of all .npy files
    sample_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    sample_ids = [f.split('.')[0] for f in sample_files]
    
    # Get true labels from labels.json
    labels_path = os.path.join(args.data_dir, 'META', 'labels.json')
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)['CODE_GROUP']
    
    # Extract true classes for each sample and convert from 1-based to 0-based indexing
    true_classes = [int(labels_data.get(str(pid), 0)) - 1 for pid in sample_ids]
    
    # Prepare results storage
    predicted_classes = []
    confidences = []
    probabilities = []
    
    # Process all samples
    print(f"\nProcessing {len(sample_ids)} samples...")
    for sample_id in tqdm(sample_ids):
        try:
            # Load and process sample
            data_array, dates = load_gee_sample(data_path, meta_path, sample_id)
            
            # Run inference
            result = inferencer.predict(data_array, dates)
            
            # Store results
            predicted_classes.append(result['predicted_class'])
            confidences.append(result['confidence'])
            probabilities.append(result['probabilities'])
        except Exception as e:
            print(f"\nError processing sample {sample_id}: {str(e)}")
            continue
    
    # Convert to DataFrame and save
    df = pd.DataFrame({
        'parcel_id': sample_ids,
        'true_class': [x + 1 for x in true_classes],  # Convert back to 1-based
        'predicted_class': [x + 1 for x in predicted_classes],  # Convert back to 1-based
        'confidence': confidences
    })
    
    # Add probability columns for each class
    probabilities = np.array(probabilities)
    for i in range(probabilities.shape[1]):
        df[f'prob_class_{i}'] = probabilities[:, i]
    
    df.to_csv(args.output_csv, index=False)
    
    print(f"\nSaved predictions to {args.output_csv}")
    
    # Generate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    
    # Create better labeled confusion matrix
    class_labels = [str(i) for i in range(10)]  # 0-9 labels
    img = sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd',
                     xticklabels=class_labels, yticklabels=class_labels,
                     square=True, cbar_kws={'label': ''})
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label (0-9)')
    plt.xlabel('Predicted Label (0-9)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Add gridlines
    plt.grid(False)
    
    # Save high-resolution figure
    plt.tight_layout()
    output_dir = os.path.dirname(args.output_csv)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print classification report (using 0-based classes for sklearn metrics)
    print("\nClass-wise Performance Analysis:")
    report = classification_report(true_classes, predicted_classes, 
                                 labels=list(range(10)),
                                 target_names=[str(i+1) for i in range(10)],
                                 zero_division=0,
                                 output_dict=True)
    
    # Calculate mean confidence per class
    class_conf = {}
    for i in range(1, 11):  # 1-based classes
        mask = df['predicted_class'] == i
        if mask.any():
            class_conf[i] = df[mask]['confidence'].mean()
        else:
            class_conf[i] = 0.0
    
    # Print class-wise metrics
    print("\nClass-wise Metrics (sorted by F1-score):")
    print("Class   F1-score  Precision  Recall  Avg.Conf  Support")
    print("-" * 55)
    
    # Get metrics for each class and sort by F1-score
    class_metrics = []
    for i in range(10):
        metrics = report[str(i+1)]
        class_metrics.append({
            'class': i+1,
            'f1': metrics['f1-score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'support': metrics['support'],
            'conf': class_conf[i+1]
        })
    
    # Sort by F1-score in descending order
    class_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    for metrics in class_metrics:
        if metrics['support'] > 0:  # Only show classes that have samples
            print(f"{metrics['class']:5d}   {metrics['f1']:.3f}    {metrics['precision']:.3f}     {metrics['recall']:.3f}   {metrics['conf']:.3f}    {int(metrics['support'])}")
    
    # Create performance summary text file
    accuracy = report['accuracy']
    mean_conf = df['confidence'].mean()
    
    # Get high confidence metrics
    high_conf_mask = df['confidence'] >= inferencer.config['confidence_threshold']
    high_conf_df = df[high_conf_mask]
    high_conf = len(high_conf_df)
    med_conf = ((df['confidence'] >= 0.50) & (df['confidence'] < 0.80)).sum()
    low_conf = (df['confidence'] < 0.50).sum()
    
    # Get class distribution
    class_counts = {}
    for i in range(10):
        class_counts[i] = (df['true_class'] == i+1).sum()
    
    # Create summary text
    summary_text = f"Overall Accuracy: {accuracy:.4f}\n\n"
    summary_text += "Classification Report (High Confidence Predictions >= 0.80, Classes 0-9):\n"
    summary_text += classification_report(
        [x - 1 for x in high_conf_df['true_class']],
        [x - 1 for x in high_conf_df['predicted_class']],
        labels=list(range(10)),
        target_names=[str(i) for i in range(10)],
        zero_division=0)
    
    summary_text += "\nPrediction Summary:\n"
    summary_text += f"Total samples processed: {len(df)}\n\n"
    
    summary_text += "Confidence distribution:\n"
    summary_text += f"High confidence (>=0.80): {high_conf} samples ({high_conf/len(df)*100:.2f}%)\n"
    summary_text += f"Medium confidence (0.50-0.80): {med_conf} samples ({med_conf/len(df)*100:.2f}%)\n"
    summary_text += f"Low confidence (<0.50): {low_conf} samples ({low_conf/len(df)*100:.2f}%)\n\n"
    
    summary_text += "Class distribution (0-9):\n"
    for i in range(10):
        summary_text += f"Class {i}: {class_counts[i]} samples\n"
    
    summary_text += f"\nMean confidence: {mean_conf:.4f}"
    
    # Save summary to file
    summary_path = os.path.join(output_dir, 'model_performance_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\nResults saved to {output_dir}")
    
    # Analyze high confidence predictions by class
    print("\nHigh Confidence (>=0.80) Performance by Class:")
    high_conf_mask = df['confidence'] >= 0.80
    high_conf_report = classification_report(
        [x - 1 for x in df[high_conf_mask]['true_class']],
        [x - 1 for x in df[high_conf_mask]['predicted_class']],
        labels=list(range(10)),
        target_names=[str(i+1) for i in range(10)],
        zero_division=0,
        output_dict=True
    )
    
    print("\nClass   F1-score  Precision  Recall  Support")
    print("-" * 45)
    
    # Sort classes by F1-score for high confidence predictions
    high_conf_metrics = []
    for i in range(10):
        metrics = high_conf_report[str(i+1)]
        if metrics['support'] > 0:
            high_conf_metrics.append({
                'class': i+1,
                'f1': metrics['f1-score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'support': metrics['support']
            })
    
    high_conf_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    for metrics in high_conf_metrics:
        print(f"{metrics['class']:5d}   {metrics['f1']:.3f}    {metrics['precision']:.3f}     {metrics['recall']:.3f}    {int(metrics['support'])}")

    # Analyze medium confidence predictions by class
    print("\nMedium Confidence (0.50-0.80) Performance by Class:")
    med_conf_mask = (df['confidence'] >= 0.50) & (df['confidence'] < 0.80)
    med_conf_report = classification_report(
        [x - 1 for x in df[med_conf_mask]['true_class']],
        [x - 1 for x in df[med_conf_mask]['predicted_class']],
        labels=list(range(10)),
        target_names=[str(i+1) for i in range(10)],
        zero_division=0,
        output_dict=True
    )
    
    print("\nClass   F1-score  Precision  Recall  Support")
    print("-" * 45)
    
    # Sort classes by F1-score for medium confidence predictions
    med_conf_metrics = []
    for i in range(10):
        metrics = med_conf_report[str(i+1)]
        if metrics['support'] > 0:
            med_conf_metrics.append({
                'class': i+1,
                'f1': metrics['f1-score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'support': metrics['support']
            })
    
    med_conf_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    for metrics in med_conf_metrics:
        print(f"{metrics['class']:5d}   {metrics['f1']:.3f}    {metrics['precision']:.3f}     {metrics['recall']:.3f}    {int(metrics['support'])}")

    # Analyze low confidence predictions by class
    print("\nLow Confidence (<0.50) Performance by Class:")
    low_conf_mask = df['confidence'] < 0.50
    low_conf_report = classification_report(
        [x - 1 for x in df[low_conf_mask]['true_class']],
        [x - 1 for x in df[low_conf_mask]['predicted_class']],
        labels=list(range(10)),
        target_names=[str(i+1) for i in range(10)],
        zero_division=0,
        output_dict=True
    )
    
    print("\nClass   F1-score  Precision  Recall  Support")
    print("-" * 45)
    
    # Sort classes by F1-score for low confidence predictions
    low_conf_metrics = []
    for i in range(10):
        metrics = low_conf_report[str(i+1)]
        if metrics['support'] > 0:
            low_conf_metrics.append({
                'class': i+1,
                'f1': metrics['f1-score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'support': metrics['support']
            })
    
    low_conf_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    for metrics in low_conf_metrics:
        print(f"{metrics['class']:5d}   {metrics['f1']:.3f}    {metrics['precision']:.3f}     {metrics['recall']:.3f}    {int(metrics['support'])}")

    # Calculate accuracy
    accuracy = (df['true_class'] == df['predicted_class']).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f}")
