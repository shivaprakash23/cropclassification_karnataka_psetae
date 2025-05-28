import argparse
import subprocess
import os
import json
import time
import sys

def log_trial(trial_num, params, status, error_msg=None, val_acc=None):
    log_path = os.path.join(os.getcwd(), 'hyperparameter_log.txt')
    with open(log_path, 'a') as f:
        f.write(f"Trial {trial_num}:\n")
        f.write(f"  Params: {json.dumps(params)}\n")
        f.write(f"  Status: {status}\n")
        if val_acc is not None:
            f.write(f"  Validation Accuracy: {val_acc}\n")
        if error_msg:
            f.write(f"  Error: {error_msg}\n")
        f.write("-"*40 + "\n")


def parse_list_param(param_str):
    """Parse a string representation of a list into an actual list of integers."""
    try:
        # Remove brackets and split by comma
        param_str = param_str.strip('[]')
        return [int(x.strip()) for x in param_str.split(',')]
    except:
        return None

def run_training(params, base_dir, trial_num):
    """Run a single training trial with the given parameters"""
    
    # Check n_head, d_k, and d_model compatibility (model hidden dim is 128)
    n_head = int(params.get('n_head', 4))
    d_k = int(params.get('d_k', 32))
    d_model = d_k  # d_model must equal d_k
    model_hidden_dim = 128
    
    # Check compatibility conditions
    if n_head * d_k != model_hidden_dim:
        msg = f"Incompatible n_head*d_k: {n_head}*{d_k} != {model_hidden_dim} (trial {trial_num})"
        print(f"[ERROR] {msg}")
        log_trial(trial_num, params, status="FAILED", error_msg=msg)
        return {"trial": trial_num, "status": "FAILED", "error": msg}
    
    # Fix n_head and d_k as requested
    params['n_head'] = 4
    params['d_k'] = 32
    d_model = params['d_k']  # d_model must equal d_k
    
    # Configure temporal attention encoder dimensions
    # 1. First layer of mlp3 must be n_head * d_model (4 * 32 = 128)
    # 2. Last layer of mlp3 must be d_model (32)
    first_layer = params['n_head'] * d_model  # 4 * 32 = 128
    mlp3 = [first_layer, model_hidden_dim, d_model]  # [128, 128, 32]
    
    # Set all required parameters
    params['d_model'] = d_model
    params['mlp3'] = f'[{first_layer},{model_hidden_dim},{d_model}]'  # [128,128,32]
    params['mlp1'] = '[8,32,64]'      # Input processing
    params['mlp2'] = '[128,128]'       # PSE dimensions
    params['mlp4'] = '[128,64,32,10]'  # Classifier dimensions (10 classes)
    params['input_dim'] = 8           # hlsl30 bands
    params['positions'] = 'order'       # Position encoding type
    params['T'] = 366                  # Temporal period
    params['lms'] = 210                # Max sequence length
    params['num_classes'] = 10         # Number of crop classes
    params['pooling'] = 'mean_std'     # PSE pooling strategy
    params['geomfeat'] = 1             # Use geometric features
    params['device'] = 'cpu'           # Run on CPU

    # Create result directory
    trial_dir = os.path.join(base_dir, f"trial_{trial_num}")
    os.makedirs(trial_dir, exist_ok=True)
    
    print(f"\n[DEBUG] Created trial directory: {trial_dir}")
    
    # Calculate mlp1 and mlp2 dimensions correctly for S2 data
    # For hlsl30 data, input_dim is 8 (8 bands)
    input_dim = 8
    
    # Define mlp1 dimensions
    mlp1 = [input_dim, 32, 64]  # Starting with input_dim (10 for S2)
    mlp1_str = f"[{','.join(map(str, mlp1))}]"
    
    # Set mlp2 based on pooling to satisfy PSE model constraints
    pooling = params.get('pooling', 'mean_std')  # Default to mean_std if not specified
    
    # Set mlp2 dimensions to match train.py default
    mlp2_str = '[128,128]'  # Use the same mlp2 dimensions as in train.py default
        
    print(f"[DEBUG] Using input_dim={input_dim}, mlp1={mlp1_str}")
    print(f"[DEBUG] Using pooling={pooling}, mlp2={mlp2_str}, with_geomfeat=0")
    
    # Build command for train.py
    cmd = [
        'python', 'train.py',
        '--sensor', 'hlsl30',
        '--dataset_folder', params['dataset_folder'],
        '--val_folder', params['val_folder'],
        '--test_folder', params['test_folder'],
        '--epochs', str(params['epochs']),
        '--lr', str(params['lr']),
        '--batch_size', str(params['batch_size']),
        '--gamma', str(params['gamma']),
        '--npixel', str(params['npixel']),
        '--minimum_sampling', str(params['minimum_sampling']),
        '--dropout', str(params['dropout']),
        '--input_dim', str(input_dim),
        '--mlp1', mlp1_str,
        '--pooling', pooling,
        '--mlp2', mlp2_str,
        '--geomfeat', '0',
        '--n_head', str(params['n_head']),
        '--d_k', str(params['d_k']),
        '--mlp3', '[512,128,128]',  # Default from train.py
        '--mlp4', '[128,64,32,10]',  # Default from train.py
        '--T', '366',  # Default from train.py
        '--positions', 'order',  # Using order for positional encoding
        '--lms', '210',  # Maximum sequence length for positional encoding
        '--num_classes', '10',  # Default from train.py
        '--device', 'cpu',  # Force CPU usage or cuda if GPU is available
        '--res_dir', trial_dir
    ]
    
    # Verify data folders exist
    for folder_param in ['dataset_folder', 'val_folder', 'test_folder']:
        folder_path = params[folder_param]
        if not os.path.exists(folder_path):
            print(f"[ERROR] {folder_param} path does not exist: {folder_path}")
        else:
            # Check if DATA and META subdirectories exist
            data_dir = os.path.join(folder_path, 'DATA')
            meta_dir = os.path.join(folder_path, 'META')
            if not os.path.exists(data_dir) or not os.path.exists(meta_dir):
                print(f"[ERROR] Missing DATA or META directory in {folder_path}")
            else:
                data_files = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
                print(f"[DEBUG] {folder_param} contains {data_files} data files")
    
    # Print trial info
    print(f"\nTrial {trial_num}:")
    print(f"Parameters: lr={params['lr']}, batch_size={params['batch_size']}")
    print(f"npixel={params['npixel']}, dropout={params['dropout']}")
    print(f"n_head={params['n_head']}, d_k={params['d_k']}")
    
    # Run training
    try:
        # Run the command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout = result.stdout
        stderr = result.stderr
        val_acc = None
        status = "SUCCESS"
        error_msg = None
        if result.returncode != 0:
            print(f"[ERROR] Trial {trial_num} failed.\n{stderr}")
            status = "FAILED"
            error_msg = stderr
        # Parse validation accuracy from stdout if possible
        for line in stdout.splitlines():
            if 'validation accuracy' in line.lower():
                try:
                    val_acc = float(line.strip().split(':')[-1].replace('%','').strip())
                except:
                    pass
        log_trial(trial_num, params, status=status, error_msg=error_msg, val_acc=val_acc)
        return {'trial': trial_num, 'status': status, 'val_acc': val_acc, 'error': error_msg}
    except Exception as e:
        print(f"[ERROR] Exception in trial {trial_num}: {e}")
        log_trial(trial_num, params, status="FAILED", error_msg=str(e))
        return {'trial': trial_num, 'status': 'FAILED', 'error': str(e)}
        if os.path.exists(trial_dir):
            print(f"[DEBUG] Contents of {trial_dir}:")
            for item in os.listdir(trial_dir):
                print(f"  {item}")
            return None
            
        with open(log_path) as f:
            logs = json.load(f)
        last_epoch = max(logs.keys(), key=int)
        val_acc = logs[last_epoch]['val_accuracy']
        print(f"Trial {trial_num} completed with validation accuracy: {val_acc:.2f}%")
        return {
            'params': params,
            'accuracy': val_acc,
            'trial': trial_num
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error reading results for trial {trial_num}: {str(e)}")
        # Print more details about the error
        print(f"[DEBUG] Error details: {type(e).__name__}: {e}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', required=True, help='Path to dataset folder')
    parser.add_argument('--val_folder', required=True, help='Path to validation folder')
    parser.add_argument('--test_folder', required=True, help='Path to test folder')
    parser.add_argument('--res_dir', default='hypertuning_results_hlsl30', help='Path to results directory')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('--pooling', default='mean_std', help='Pooling type')
    parser.add_argument('--geomfeat', default=1, type=int, help='Use geometric features (1) or not (0)')
    args = parser.parse_args()
    
    print("\n[DEBUG] Starting hyperparameter tuning with arguments:")
    print(f"  dataset_folder: {args.dataset_folder}")
    print(f"  val_folder: {args.val_folder}")
    print(f"  test_folder: {args.test_folder}")
    print(f"  res_dir: {args.res_dir}")
    print(f"  epochs: {args.epochs}")
    print(f"  pooling: {args.pooling}")
    print(f"  geomfeat: {args.geomfeat}")
    
    # Create results directory
    os.makedirs(args.res_dir, exist_ok=True)
    print(f"[DEBUG] Created results directory: {os.path.abspath(args.res_dir)}")
    
    # Define parameter grid (simplified)
    param_grid = [
        # Trial 1: Best for Feature Learning
        {
            'lr': 0.0001,
            'batch_size': 32,
            'gamma': 1.0,
            'npixel': 8,
            'minimum_sampling': 8,
            'dropout': 0.2,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 2: Best for Regularization
        {
            'lr': 0.0001,
            'batch_size': 32,
            'gamma': 0.5,
            'npixel': 8,
            'minimum_sampling': 8,
            'dropout': 0.3,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 3: Best for Fast Learning
        {
            'lr': 0.001,
            'batch_size': 64,
            'gamma': 1.0,
            'npixel': 16,
            'minimum_sampling': 8,
            'dropout': 0.1,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 4: Balanced Learning
        {
            'lr': 0.0001,
            'batch_size': 32,
            'gamma': 0.5,
            'npixel': 16,
            'minimum_sampling': 8,
            'dropout': 0.2,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 5: High Resolution Focus
        {
            'lr': 0.0001,
            'batch_size': 64,
            'gamma': 1.0,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.3,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 6: Temporal Pattern Focus
        {
            'lr': 0.0001,
            'batch_size': 32,
            'gamma': 1.0,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.2,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 7: Robust Learning
        {
            'lr': 0.0001,
            'batch_size': 64,
            'gamma': 0.5,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.3,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 8: Fast Convergence
        {
            'lr': 0.001,
            'batch_size': 32,
            'gamma': 0.5,
            'npixel': 64,
            'minimum_sampling': 8,
            'dropout': 0.2,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 9: Memory Efficient
        {
            'lr': 0.0001,
            'batch_size': 64,
            'gamma': 1.0,
            'npixel': 32,
            'minimum_sampling': 4,
            'dropout': 0.2,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 10: Aggressive Learning
        {
            'lr': 0.001,
            'batch_size': 64,
            'gamma': 1.0,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.1,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 11: Conservative Learning
        {
            'lr': 0.0001,
            'batch_size': 32,
            'gamma': 0.5,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.3,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 12: Minimal Resource
        {
            'lr': 0.001,
            'batch_size': 64,
            'gamma': 0.5,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.2,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 13: Config A (Stable, Large Batch)
        {
            'lr': 0.001,
            'batch_size': 128,
            'gamma': 0.97,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.05,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },
        # Trial 14: Config C (Stable, Small LR)
        {
            'lr': 0.0005,
            'batch_size': 128,
            'gamma': 1.0,
            'npixel': 32,
            'minimum_sampling': 8,
            'dropout': 0.05,
            'pooling': 'mean_std',
            'geomfeat': 1,
            'n_head': 4,
            'd_k': 32,
            'input_dim': 8,
            'mlp1': '[8,32,64]',
            'mlp2': '[128,128]',
            'positions': 'order',
            'T': 366,
            'lms': 210,
            'epochs': args.epochs,
            'dataset_folder': args.dataset_folder,
            'val_folder': args.val_folder,
            'test_folder': args.test_folder
        },

    ]
    
    # Run trials
    results = []
    for i, params in enumerate(param_grid):
        result = run_training(params, args.res_dir, i)
        if result:
            # Read overall.json to get validation accuracy
            overall_json_path = os.path.join(args.res_dir, f'trial_{i}', 'overall.json')
            if os.path.exists(overall_json_path):
                try:
                    with open(overall_json_path, 'r') as f:
                        overall_metrics = json.load(f)
                        result['val_acc'] = overall_metrics.get('overall_accuracy', None)
                except Exception as e:
                    print(f'Error reading overall.json for trial {i}: {e}')
                    result['val_acc'] = None
            results.append(result)
    
    # Find best result
    if results:
        best_result = max(results, key=lambda x: x['val_acc'] if x['val_acc'] is not None else float('-inf'))
        
        # Save best parameters
        with open(os.path.join(args.res_dir, 'best_params.json'), 'w') as f:
            json.dump(best_result, f, indent=2)
        
        print(f"\nBest trial: #{best_result['trial']}")
        print(f"Best validation accuracy: {best_result['val_acc']:.2f}%")
        print("Best hyperparameters:")
        for param, value in best_result['params'].items():
            if param not in ['dataset_folder', 'val_folder', 'test_folder', 'epochs']:
                print(f"  {param}: {value}")
    else:   
        print("No successful trials.")

if __name__ == '__main__':
    main()
