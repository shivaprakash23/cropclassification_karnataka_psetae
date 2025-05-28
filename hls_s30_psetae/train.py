"""
Training script for HLS s30 PSETAE model
"""

import os
import json
import time
import pprint
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchnet as tnt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import PixelSetData, PixelSetData_preloaded, custom_collate_fn
from models.stclassifier import PseTae
from learning.weight_init import weight_init
from learning.focal_loss import FocalLoss
from learning.metrics import confusion_matrix_analysis, mIou

def recursive_todevice(x, device):
    """Recursively moves a nested structure of tensors to the specified device."""
    if isinstance(x, (list, tuple)):
        return [recursive_todevice(x_i, device) for x_i in x]
    else:
        return x.to(device)

def prepare_output(config):
    """Create output directories for results."""
    os.makedirs(config['res_dir'], exist_ok=True)

def checkpoint(log, config):
    """Save training log checkpoint."""
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'w') as f:
        json.dump(log, f)

def save_results(metrics, conf_mat, config, y_true, y_pred, probs):
    """Save test results and generate visualizations."""
    # Save metrics
    with open(os.path.join(config['res_dir'], 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    # Save overall metrics
    overall = {
        'overall_accuracy': float(metrics['test_acc']),
        'mean_IoU': float(metrics['test_iou']),
        'loss': float(metrics['test_loss'])
    }
    with open(os.path.join(config['res_dir'], 'overall.json'), 'w') as f:
        json.dump(overall, f, indent=4)
    
    # Save confusion matrix
    np.save(os.path.join(config['res_dir'], 'confusion_matrix.npy'), conf_mat)
    
    # Convert y_true and y_pred to integer arrays if they contain elements
    if len(y_true) > 0:
        try:
            y_true_int = np.array([int(y) if not isinstance(y, int) else y for y in y_true])
            np.save(os.path.join(config['res_dir'], 'y_true.npy'), y_true_int)
        except Exception as e:
            print(f"Warning: Could not save y_true as integers: {e}")
            np.save(os.path.join(config['res_dir'], 'y_true.npy'), np.array(y_true, dtype=object))
    
    if len(y_pred) > 0:
        try:
            y_pred_int = np.array([int(y) if not isinstance(y, int) else y for y in y_pred])
            np.save(os.path.join(config['res_dir'], 'y_pred.npy'), y_pred_int)
        except Exception as e:
            print(f"Warning: Could not save y_pred as integers: {e}")
            np.save(os.path.join(config['res_dir'], 'y_pred.npy'), np.array(y_pred, dtype=object))
    
    # Save probabilities
    if len(probs) > 0:
        np.save(os.path.join(config['res_dir'], 'probabilities.npy'), probs)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Reds')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(config['res_dir'], 'conf_mat_picture.png'))
    plt.close()
    
    # Calculate and save per-class metrics
    try:
        per_class, overall = confusion_matrix_analysis(conf_mat)
    except Exception as e:
        print(f"Warning: Could not calculate detailed metrics: {e}")
        per_class, overall = {}, {}
    with open(os.path.join(config['res_dir'], 'detailed_metrics.json'), 'w') as f:
        json.dump({'per_class': per_class, 'overall': overall}, f, indent=2)

def plot_metrics(config):
    """Plot training and validation metrics."""
    with open(os.path.join(config['res_dir'], 'trainlog.json'), 'r') as f:
        log = json.load(f)

    epochs = sorted([int(e) for e in log.keys()])
    train_loss = [log[str(e)]['loss'] for e in epochs]
    val_loss = [log[str(e)]['val_loss'] for e in epochs]
    train_acc = [log[str(e)]['acc'] for e in epochs]  # Already in 0-100 range
    val_acc = [log[str(e)]['val_acc'] for e in epochs]  # Already in 0-100 range

    # Plot Loss Graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, '-', color='#1f77b4', label='training loss')
    plt.plot(epochs, val_loss, '-', color='#ff7f0e', label='validation loss')
    plt.title('monitoring metrics - loss')
    plt.xlabel('epoch')
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config['res_dir'], 'loss_graph.png'))
    plt.close()

    # Plot Accuracy Graph
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, '-', color='#1f77b4', label='training accuracy')
    plt.plot(epochs, val_acc, '-', color='#ff7f0e', label='validation accuracy')
    plt.title('monitoring metrics - accuracy')
    plt.xlabel('epoch')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config['res_dir'], 'accuracy_graph.png'))
    plt.close()
    plt.close()


def train_epoch(model, optimizer, criterion, data_loader, device, config, scheduler=None):
    """Train model for one epoch.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        data_loader: Training dataloader
        device: Device to use
        config: Configuration dictionary
        
    Returns:
        dict: Epoch metrics including loss and accuracy
    """
    model.train()
    
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    
    for i, ((x, mask), y, dates) in enumerate(data_loader):
        x = recursive_todevice(x, device)
        y = y.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        prediction = model((x, mask), dates)
        
        # Calculate loss
        loss = criterion(prediction, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update learning rate if scheduler provided
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        acc_meter.add(prediction.detach(), y)
        loss_meter.add(loss.item())
        
        # Print progress
        if (i + 1) % config['display_step'] == 0:
            print('  Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'.format(
                i + 1,
                len(data_loader),
                loss_meter.value()[0],
                acc_meter.value()[0]
            ))
    
    return {
        'loss': loss_meter.value()[0],
        'acc': acc_meter.value()[0]
    }


def evaluation(model, criterion, loader, device, config, mode='val'):
    """Evaluate model on validation or test set.
    
    Args:
        model: Model to evaluate
        criterion: Loss function
        loader: Data loader
        device: Device to use
        config: Configuration dictionary
        mode: Either 'val' or 'test'
        
    Returns:
        dict: Metrics dictionary
        np.ndarray: Confusion matrix (only for test mode)
        list: True labels (only for test mode)
        list: Predicted labels (only for test mode)
        list: Prediction probabilities (only for test mode)
    """
    y_true = []
    y_pred = []
    probs = []

    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()

    for (x, y, dates) in loader:
        # Store true labels as Python integers
        y_true.extend(list(map(int, y)))
        
        x = recursive_todevice(x, device)
        y = y.to(device)

        with torch.no_grad():
            prediction = model(x, dates)
            loss = criterion(prediction, y)

        acc_meter.add(prediction.detach(), y)
        loss_meter.add(loss.item())

        # Get predictions and probabilities
        y_p = prediction.detach().argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        probs.extend(list(prediction.detach().cpu().numpy()))

    # Calculate metrics
    prefix = mode + '_' if mode else ''
    iou_score = mIou(y_true, y_pred, config['num_classes'])
    metrics = {
        f'{mode}_loss': loss_meter.value()[0],
        f'{mode}_acc': acc_meter.value()[0],
        f'{mode}_iou': iou_score
    }

    if mode == 'val':
        return metrics
    elif mode == 'test':
        from sklearn.metrics import confusion_matrix
        return metrics, confusion_matrix(y_true, y_pred, labels=list(range(config['num_classes']))), y_true, y_pred, probs


def get_pse(folder, config):
    """Get dataset for train/val/test.
    Args:
        folder: Path to data directory
        config: Configuration dictionary
    """
    sub_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    config['n_classes'] = len(sub_classes)
    if config['preload']:
        dt = PixelSetData_preloaded(folder, labels='CODE_GROUP', npixel=config['npixel'],
                          sub_classes=sub_classes,
                          norm=None,
                          minimum_sampling=config['minimum_sampling'],
                          extra_feature='geomfeat' if config['geomfeat'] else None,  
                          jitter=None)
    else:
        dt = PixelSetData(folder, labels='CODE_GROUP', npixel=config['npixel'],
                          sub_classes=sub_classes,
                          norm=None,
                          minimum_sampling=config['minimum_sampling'],
                          extra_feature='geomfeat' if config['geomfeat'] else None, 
                          jitter=None)
    return dt


def train(config):
    """
    Main training function
    
    Args:
        config: Configuration dictionary
    """
    # Create results 
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    prepare_output(config)

    extra = 'geomfeat' if config['geomfeat'] else None
    device = torch.device(config['device'])

    # Get datasets
    print('Loading training dataset...')
    train_dt = get_pse(config['dataset_folder'], config)
    print(f'Training dataset loaded with {len(train_dt)} samples')
    
    print('Loading validation dataset...')
    val_dt = get_pse(config['val_folder'], config) if config['val_folder'] else None
    if val_dt:
        print(f'Validation dataset loaded with {len(val_dt)} samples')
    
    print('Loading test dataset...')
    test_dt = get_pse(config['test_folder'], config) if config['test_folder'] else None
    if test_dt:
        print(f'Test dataset loaded with {len(test_dt)} samples')

    # Create dataloaders with custom collate function
    trn_loader = DataLoader(train_dt, batch_size=config['batch_size'], shuffle=True,
                            num_workers=config['num_workers'], pin_memory=True,
                            collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dt, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['num_workers'],
                          collate_fn=custom_collate_fn) if val_dt else None
    test_loader = DataLoader(test_dt, batch_size=config['batch_size'], shuffle=False,
                           num_workers=config['num_workers'],
                           collate_fn=custom_collate_fn) if test_dt else None

    print('Train {}, Val {}, Test {}'.format(
        len(trn_loader.dataset), 
        len(val_loader.dataset) if val_loader else 0,
        len(test_loader.dataset) if test_loader else 0
    ))

    model_config = dict(
        input_dim=config['input_dim'], 
        mlp1=config['mlp1'], 
        pooling=config['pooling'],
        mlp2=config['mlp2'], 
        n_head=config['n_head'], 
        d_k=config['d_k'], 
        mlp3=config['mlp3'],
        dropout=config['dropout'], 
        T=config['T'], 
        len_max_seq=config['lms'],
        positions=None,
        mlp4=config['mlp4'],
        with_extra=False
    )

    if config['geomfeat']:
        model_config.update(with_extra=True, extra_size=4)

    model = PseTae(**model_config)

    # Initialize model and move to device
    model = model.to(device)
    model.apply(weight_init)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = FocalLoss(config['gamma'])

    trainlog = {}
    best_mIoU = 0

    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        print('\n' + '='*50)
        print('Starting EPOCH {}/{}'.format(epoch, config['epochs']))
        print('='*50)

        print('\nTraining phase...')
        model.train()
        train_metrics = train_epoch(model, optimizer, criterion, trn_loader, device=device, config=config)
        train_loss = train_metrics['loss']
        train_acc = train_metrics['acc']
        print('Training Results:')
        print('  - Loss: {:.4f}'.format(train_loss))
        print('  - Accuracy: {:.2f}'.format(train_acc))

        print('\nValidation phase...')
        model.eval()
        val_metrics = evaluation(model, criterion, val_loader, device=device, config=config, mode='val')
        val_loss = val_metrics['val_loss']
        val_acc = val_metrics['val_acc']
        val_iou = val_metrics['val_iou']
        print('Validation Results:')
        print('  - Loss: {:.4f}'.format(val_loss))
        print('  - Accuracy: {:.2f}'.format(val_acc))
        print('  - IoU Score: {:.4f}'.format(val_iou))

        # Store training and validation metrics separately
        trainlog[epoch] = {
            'loss': train_loss,
            'acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_iou': val_iou
        }
        checkpoint(trainlog, config)

        if val_iou >= best_mIoU:
            best_mIoU = val_iou
            torch.save(
                {
                    'epoch': epoch, 
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                os.path.join(config['res_dir'], 'model.pth.tar')
            )

    print('Testing best epoch . . .')
    model.load_state_dict(
        torch.load(os.path.join(config['res_dir'], 'model.pth.tar'))['state_dict']
    )
    model.eval()

    # Run test evaluation
    test_metrics, conf_mat, y_true, y_pred, probs = evaluation(
        model, criterion, test_loader, device=device, mode='test', config=config
    )

    print('Test Loss {:.4f}, Test Acc {:.2f}, Test IoU {:.4f}'.format(
        test_metrics['test_loss'],
        test_metrics['test_acc'],
        test_metrics['test_iou']))
                                                             
    save_results(test_metrics, conf_mat, config, y_true, y_pred, probs)

    plot_metrics(config)

    # Calculate overall performance metrics
    overall_performance = {
        'best_val_iou': best_mIoU,
        'final_test_metrics': test_metrics
    }
    
    print("\nTraining completed!")


if __name__ == '__main__':
    start = datetime.now()

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')

    # set-up data loader folders -----------------------------
    parser.add_argument('--dataset_folder2', default=None, type=str,
                        help='Path to second train folder to concat with first initial loader.')
    parser.add_argument('--val_folder', default=None, type=str,
                        help='Path to the validation folder.')
    parser.add_argument('--test_folder', default=None, type=str,
                        help='Path to the test folder.')

    # sensor argument to s1/s2
    parser.add_argument('--sensor', default='HLSS30', type=str,
                        help='Type of mission data to train e.g.HLSS30')
    parser.add_argument('--minimum_sampling', default=22, type=int,
                        help='minimum time series length to sample for HLSS30')

    parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=7, type=int, help='Random seed')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)

    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=16, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[128,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=0, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## TAE
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=366, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='order', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=210, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=10, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 10]', type=str, help='Number of neurons in the layers of MLP4')

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    train(config)

    print('total elapsed time is --->', datetime.now() - start)
