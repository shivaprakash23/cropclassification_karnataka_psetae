import torch
import torch.utils.data as data
import numpy as np
import os
import json
import torchnet as tnt
import argparse
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import pandas as pd

import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from models.stclassifier_fusion import PseTae
from dataset_fusion import PixelSetData
from learning.metrics import mIou, confusion_matrix_analysis

class InferenceFusion:
    def __init__(self, config):
        self.device = torch.device(config['device'])
        self.config = config
        self.batch_size = config.get('batch_size', 1)  # Default to 1 for sequential processing
        self.s1_data_folder = config['s1_data_folder']
        self.s2_data_folder = config['s2_data_folder']
        self.model = self.load_model()
        
    def load_model(self):
        # Initialize model with training configuration
        model_config = dict(
            input_dim=self.config['input_dim'],
            mlp1=self.config['mlp1'],
            pooling=self.config['pooling'],
            mlp2=self.config['mlp2'],
            n_head=self.config['n_head'],
            d_k=self.config['d_k'],
            mlp3=self.config['mlp3'],
            dropout=self.config['dropout'],
            T=self.config['T'],
            len_max_seq=self.config['lms'],
            positions=None,
            fusion_type=self.config['fusion_type'],
            mlp4=self.config['mlp4']
        )
        
        if self.config['geomfeat']:
            model_config.update(with_extra=True, extra_size=4)
        else:
            model_config.update(with_extra=False, extra_size=None)
            
        model = PseTae(**model_config)
        
        # Load weights
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        model.eval()
        return model

    def prepare_datasets(self):
        """Prepare datasets for S1 and S2 data"""
        # Basic dataset configuration
        dataset_config = {
            'labels': 'CODE_GROUP',
            'npixel': self.config['npixel'],
            'sub_classes': None,
            'norm': None,
            'minimum_sampling': self.config['minimum_sampling'],
            'fusion_type': self.config['fusion_type'],
            'return_id': True
        }
        
        # Create S1 dataset
        s1_dataset = PixelSetData(
            folder=self.s1_data_folder,
            **dataset_config
        )

        # Create S2 dataset
        s2_dataset = PixelSetData(
            folder=self.s2_data_folder,
            **dataset_config
        )

        return s1_dataset, s2_dataset

    def prepare_dataloader(self, dataset):
        """Prepare data loader for the given dataset"""
        # Create data loader
        data_loader = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.custom_collate,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        return data_loader

    def recursive_todevice(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, tuple):
            return tuple(self.recursive_todevice(item, device) for item in x)
        elif isinstance(x, list):
            return [self.recursive_todevice(item, device) for item in x]
        else:
            return x

    def run_inference(self):
        print('Starting inference...')
        # Get datasets
        s1_dataset, s2_dataset = self.prepare_datasets()
        print(f'Found {len(s1_dataset)} samples for inference')

        # Set model to evaluation mode
        self.model.eval()

        # Store predictions and true labels
        results = {
            'y_true': [],
            'y_pred': [],
            'y_pred_proba': [],
            'sample_ids': []
        }

        # Run inference
        with torch.no_grad():
            for idx in range(len(s1_dataset)):
                # Get samples from both datasets
                s1_sample = s1_dataset[idx]
                s2_sample = s2_dataset[idx]
                
                # Extract data
                s1_data, s1_mask = s1_sample[0]
                s2_data, s2_mask = s2_sample[0]
                target = s1_sample[2]  # Use target from either dataset
                s1_dates, s2_dates = s1_sample[3]  # Get dates
                
                # Convert to tensors if needed
                if not isinstance(s1_data, torch.Tensor):
                    s1_data = torch.from_numpy(s1_data)
                    s1_mask = torch.from_numpy(s1_mask)
                if not isinstance(s2_data, torch.Tensor):
                    s2_data = torch.from_numpy(s2_data)
                    s2_mask = torch.from_numpy(s2_mask)
                if not isinstance(s1_dates, torch.Tensor):
                    s1_dates = torch.from_numpy(s1_dates)
                if not isinstance(s2_dates, torch.Tensor):
                    s2_dates = torch.from_numpy(s2_dates)
                
                # Add batch dimension if needed
                if s1_data.dim() == 3:
                    s1_data = s1_data.unsqueeze(0)
                    s1_mask = s1_mask.unsqueeze(0)
                if s2_data.dim() == 3:
                    s2_data = s2_data.unsqueeze(0)
                    s2_mask = s2_mask.unsqueeze(0)
                if s1_dates.dim() == 1:
                    s1_dates = s1_dates.unsqueeze(0)
                if s2_dates.dim() == 1:
                    s2_dates = s2_dates.unsqueeze(0)
                
                # Move to device
                s1_input = (s1_data.to(self.device), s1_mask.to(self.device))
                s2_input = (s2_data.to(self.device), s2_mask.to(self.device))
                dates = (s1_dates.to(self.device), s2_dates.to(self.device))

                # Forward pass
                out = self.model(s1_input, s2_input, dates)

                # Get predictions
                pred = torch.nn.functional.softmax(out, dim=1)
                y_p = pred.argmax(dim=1).cpu().numpy()
                
                # Store results (subtract 1 from target to match 0-based model output)
                results['y_true'].append(int(target) - 1)  # Convert from 1-based to 0-based
                results['y_pred'].append(int(y_p[0]))  # Model output is already 0-based
                results['y_pred_proba'].append(pred.cpu().numpy()[0])
                results['sample_ids'].append(s1_sample[4])

                if (idx + 1) % 10 == 0:
                    print(f'Processed {idx + 1}/{len(s1_dataset)} samples')
        
        return results
    def analyze_results(self, results):
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        y_pred_proba = np.array(results['y_pred_proba'])
        
        # Print unique classes for debugging
        print(f"\nUnique classes in y_true: {np.unique(y_true)}")
        print(f"Unique classes in y_pred: {np.unique(y_pred)}")
        
        # Calculate confusion matrix with explicit labels to ensure all classes are included
        labels = list(range(10))  # 0-9 classes
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Calculate IoU
        iou = mIou(y_true, y_pred, self.config['num_classes'])
        
        # Get confidence scores
        confidences = np.max(y_pred_proba, axis=1)
        mean_confidence = np.mean(confidences)
        
        # Split by confidence levels
        high_conf_mask = confidences >= 0.80
        med_conf_mask = (confidences >= 0.50) & (confidences < 0.80)
        low_conf_mask = confidences < 0.50
        
        # Get classification reports
        all_report = classification_report(y_true, y_pred, zero_division=0)
        high_conf_report = classification_report(
            y_true[high_conf_mask], y_pred[high_conf_mask], zero_division=0
        ) if np.any(high_conf_mask) else 'No high confidence predictions'
        
        # Get class distribution
        class_dist = {i: np.sum(y_true == i) for i in range(10)}
        
        # Prepare detailed metrics
        test_metrics = {
            'sample_id': results['sample_ids'],
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_pred_proba': [p.tolist() for p in y_pred_proba],
            'test_accuracy': float(accuracy),
            'test_f1': float(f1),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_IoU': float(iou),
            'mean_confidence': float(mean_confidence),
            'confidence_dist': {
                'high': int(np.sum(high_conf_mask)),
                'medium': int(np.sum(med_conf_mask)),
                'low': int(np.sum(low_conf_mask))
            },
            'class_distribution': class_dist,
            'classification_report': all_report,
            'high_conf_report': high_conf_report,
            'confusion_matrix': conf_mat.tolist()
        }
        
        return test_metrics, conf_mat, y_true, y_pred

    def plot_confusion_matrix(self, conf_mat, save_path):
        plt.figure(figsize=(15, 10))
        class_labels = [str(i) for i in range(10)]  # Classes 0-9
        
        # Ensure we have a complete 10x10 matrix
        if conf_mat.shape[0] < 10 or conf_mat.shape[1] < 10:
            new_conf_mat = np.zeros((10, 10), dtype=int)
            new_conf_mat[:conf_mat.shape[0], :conf_mat.shape[1]] = conf_mat
            conf_mat = new_conf_mat
        
        # Create heatmap with raw counts (no normalization)
        sns.heatmap(conf_mat,
                  annot=True,
                  fmt='d',
                  cmap='OrRd',  # Using OrRd colormap
                  xticklabels=class_labels,
                  yticklabels=class_labels,
                  linewidths=0.5,
                  square=True,  # Make cells square
                  cbar_kws={'label': ''},  # Remove colorbar label
                  vmin=0)  # Set minimum value to 0 to ensure zeros are colored
        
        # Customize appearance
        plt.title('Confusion Matrix', pad=20, size=12)
        plt.xlabel('Predicted Label (0-9)', size=10)
        plt.ylabel('True Label (0-9)', size=10)
        
        # Adjust label parameters
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
        
        # Save plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def save_results(self, test_metrics, results, conf_mat):
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate additional metrics for the report
        y_true = np.array(results['y_true'])
        y_pred = np.array(results['y_pred'])
        y_pred_proba = np.array(results['y_pred_proba'])
        
        # Calculate confidence metrics
        confidences = np.max(y_pred_proba, axis=1)
        high_conf_mask = confidences >= 0.80
        med_conf_mask = (confidences >= 0.50) & (confidences < 0.80)
        low_conf_mask = confidences < 0.50
        
        # Calculate metrics for high confidence predictions
        high_conf_report = classification_report(y_true[high_conf_mask], y_pred[high_conf_mask]) if np.any(high_conf_mask) else "No high confidence predictions"
        
        # Calculate class distribution
        class_dist = np.bincount(y_true, minlength=10)
        pred_class_dist = np.bincount(y_pred, minlength=10)
        
        # Save detailed report as text
        report_file = os.path.join(output_dir, 'detailed_report.txt')
        with open(report_file, 'w') as f:
            # Overall metrics
            f.write("=== INFERENCE RESULTS REPORT ===\n\n")
            f.write(f"Overall Accuracy: {test_metrics['test_accuracy']:.4f}\n")
            f.write(f"Mean IoU: {test_metrics['test_IoU']:.4f}\n")
            f.write(f"F1 Score: {test_metrics['test_f1']:.4f}\n")
            f.write(f"Precision: {test_metrics['test_precision']:.4f}\n")
            f.write(f"Recall: {test_metrics['test_recall']:.4f}\n\n")
            
            # Sample counts
            f.write("=== PREDICTION SUMMARY ===\n")
            f.write(f"Total samples processed: {len(y_true)}\n\n")
            
            # Confidence distribution
            f.write("=== CONFIDENCE DISTRIBUTION ===\n")
            f.write(f"High confidence (>=0.80): {np.sum(high_conf_mask)} samples ({100*np.mean(high_conf_mask):.2f}%)\n")
            f.write(f"Medium confidence (0.50-0.80): {np.sum(med_conf_mask)} samples ({100*np.mean(med_conf_mask):.2f}%)\n")
            f.write(f"Low confidence (<0.50): {np.sum(low_conf_mask)} samples ({100*np.mean(low_conf_mask):.2f}%)\n")
            f.write(f"Mean confidence: {np.mean(confidences):.4f}\n\n")
            
            # Class distribution
            f.write("=== CLASS DISTRIBUTION ===\n")
            f.write("True labels:\n")
            for i in range(10):
                f.write(f"Class {i}: {class_dist[i]} samples ({100*class_dist[i]/len(y_true):.2f}%)\n")
            f.write("\nPredicted labels:\n")
            for i in range(10):
                f.write(f"Class {i}: {pred_class_dist[i]} samples ({100*pred_class_dist[i]/len(y_pred):.2f}%)\n")
            f.write("\n")
            
            # High confidence predictions report
            f.write("=== HIGH CONFIDENCE PREDICTIONS REPORT ===\n")
            f.write("(Only for predictions with confidence >= 0.80)\n\n")
            f.write(high_conf_report)
            
        # Generate and save confusion matrix
        self.plot_confusion_matrix(conf_mat, os.path.join(output_dir, 'confusion_matrix.png'))

        # Save metrics as JSON
        metrics_dict = {
            'test_accuracy': float(test_metrics['test_accuracy']),
            'test_IoU': float(test_metrics['test_IoU']),
            'test_f1': float(test_metrics['test_f1']),
            'test_precision': float(test_metrics['test_precision']),
            'test_recall': float(test_metrics['test_recall']),
            'mean_confidence': float(np.mean(confidences)),
            'high_confidence_samples': int(np.sum(high_conf_mask)),
            'medium_confidence_samples': int(np.sum(med_conf_mask)),
            'low_confidence_samples': int(np.sum(low_conf_mask)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_dict, f, indent=4)
            
        # Save predictions
        predictions_df = pd.DataFrame({
            'sample_id': results['sample_ids'],
            'true_label': y_true,
            'predicted_label': y_pred,
            'confidence': confidences
        })
        predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model_path', type=str,
                        default='D:/Semester4/ProjectVijayapur/psetae/psetae_all5models/5_sentinel_s1_s2_fusion/multi_sensor/inference/inferencing_models/softmax/s1_s2_results_softmax_seed58/model.pth.tar',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--s1_data_folder', type=str, 
                        default='D:/Semester4/ProjectVijayapur/psetae/psetae_all5models/5_sentinel_s1_s2_fusion/multi_sensor/inference/s1_data',
                        help='Path to the S1 data folder for inference')
    parser.add_argument('--s2_data_folder', type=str, 
                        default='D:/Semester4/ProjectVijayapur/psetae/psetae_all5models/5_sentinel_s1_s2_fusion/multi_sensor/inference/s2_data',
                        help='Path to the S2 data folder for inference')
    parser.add_argument('--fusion_type', type=str, default='softmax_avg',
                        help='Type of fusion to use (e.g., softmax_avg)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--npixel', type=int, default=64,
                        help='Number of pixels to sample from each parcel')
    parser.add_argument('--minimum_sampling', type=int, default=24,
                        help='Minimum number of time steps to sample')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation (cpu/cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, 
                        default='D:/Semester4/ProjectVijayapur/psetae/psetae_all5models/5_sentinel_s1_s2_fusion/multi_sensor/inference/inference_results_softmax',
                        help='Path to save inference results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert args to config dict
    config = vars(args)
    
    # Add model architecture parameters
    model_config = {
        'input_dim': 10,
        'mlp1': [10, 32, 64],
        'pooling': 'mean_std',
        'mlp2': [128, 128],
        'geomfeat': 0,
        'n_head': 4,
        'd_k': 32,
        'mlp3': [512, 128, 128],
        'T': 366,
        'positions': 'order',
        'lms': 210,
        'dropout': 0.2,
        'mlp4': [128, 64, 32, 10]
    }
    
    # Add dataset parameters
    dataset_config = {
        'labels': 'CODE_GROUP',
        'norm': None,
        'interpolate_method': 'nn',
        'preload': False,
        'batch_size': 32,
        'display_step': 50
    }
    
    config.update(model_config)
    config.update(dataset_config)

    # Initialize inference
    inference = InferenceFusion(config)

    # Run inference
    results = inference.run_inference()

    # Analyze results
    test_metrics, conf_mat, y_true, y_pred = inference.analyze_results(results)

    # Print test metrics
    print('\nTest metrics:')
    print('Test accuracy: {:.2f}'.format(test_metrics['test_accuracy']))
    print('Test IoU: {:.2f}'.format(test_metrics['test_IoU']))

    # Save all results
    inference.save_results(test_metrics, results, conf_mat)

    print('\nInference completed successfully!')
    print(f'Results saved to {args.output_dir}')

if __name__ == '__main__':
    main()
