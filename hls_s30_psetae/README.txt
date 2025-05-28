# HLS S30 PSETAE Model - Training & Inference Guide

This document explains how to train and infer using the HLS S30 PSETAE model in this folder. It also covers hyperparameter tuning and inference workflows.

---

## 1. Training the HLS S30 Model

### Script: `train.py`

#### **How to Train**
Use the following command to train the model:

```bash
python train.py --config config.json
```

- The script expects a configuration file (JSON) specifying paths, hyperparameters, and training settings.
- Example config fields (check your code/configs for exact keys):
    - `dataset_folder`: Path to training data
    - `val_folder`: Path to validation data
    - `test_folder`: Path to test data
    - `res_dir`: Directory to save results
    - `epochs`: Number of epochs
    - `batch_size`: Batch size
    - `lr`: Learning rate
    - `npixel`, `minimum_sampling`, `geomfeat`, etc.
    - `device`: 'cuda' or 'cpu'

#### **Command-Line Example**
```bash
python train.py --config config.json
```

If your script uses direct arguments (not config file), use:
```bash
python train.py --dataset_folder path/to/train --val_folder path/to/val --test_folder path/to/test --res_dir results --epochs 20 --batch_size 32 --lr 0.001 --device cuda
```

Check your `train.py` or configs for exact argument names.

#### **Outputs**
- Trained model checkpoint (`model.pth.tar`)
- Training logs (`trainlog.json`)
- Metrics and plots (accuracy, loss, confusion matrix)
- Test results (`test_metrics.json`, `overall.json`)

---

## 2. Hyperparameter Tuning

### Script: `hyperparameter_tuning_hlss30.py`

#### **Purpose**
- Runs multiple training trials with different hyperparameters.
- Logs results and validation accuracy for each trial.

#### **How to Use**
```bash
python hyperparameter_tuning_hlss30.py --dataset_folder path/to/train --val_folder path/to/val --test_folder path/to/test --epochs 20 --res_dir hypertuning_results_hlss30
```

- The script will iterate over various hyperparameter combinations (see inside the script for which parameters are tuned: `lr`, `batch_size`, `gamma`, `npixel`, `n_head`, `d_k`, etc).
- Results are logged in `hyperparameter_log.txt` and best settings can be found by reviewing this file.

#### **When to Use**
- Use this script to find the best model configuration for your dataset.
- Recommended before final training for production or publication.

---

## 3. Inference (Prediction)

### Folder: `inference/`
#### Script: `inference_hlss30.py`

#### **How to Run Inference**
Use this script to make predictions on new data using a trained model checkpoint.

```bash
python inference/inference_hlss30.py --model_path path/to/model.pth.tar --data_dir path/to/infer_data --output_dir path/to/save_results --device cuda
```

- `--model_path`: Path to the trained model checkpoint (from training)
- `--data_dir`: Folder with `DATA` and `META` subfolders (same format as training data)
- `--output_dir`: Where to save predictions/results
- `--device`: 'cuda' or 'cpu'

#### **Outputs**
- Predicted classes, confidence scores, and evaluation metrics (if ground truth is available)
- Results are saved in the specified output directory

#### **When to Use**
- Use for batch prediction on new or test data after training is complete.
- Useful for generating submission files, validating model performance, or downstream analysis.

---

## 4. Notes & Tips
- Make sure your data is in the correct format (DATA and META folders, with .npy arrays and JSON metadata).
- For hyperparameter tuning, check compatibility of `n_head`, `d_k`, and model hidden dimensions as described in the script.
- Review logs and plots in the results directory for model diagnostics.

---

For any further details, check the docstrings and comments within each script, or ask for specific usage examples.
