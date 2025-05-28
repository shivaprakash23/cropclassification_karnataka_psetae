# Sentinel PSETAE - Training, Hyperparameter Tuning, and Inference Guide

This document explains how to use the single_sensor and multi_sensor workflows for Sentinel-1 and Sentinel-2 based crop classification using the PSETAE architecture. It covers training, hyperparameter tuning, and inference for both setups.

---

## 1. Single Sensor Workflow (sentinel_psetae/single_sensor)

### Training
- **Script:** `train.py`
- **How to Run:**
    ```bash
    python train.py --config config.json
    ```
    - The script expects a configuration file specifying dataset paths, hyperparameters, and training settings.
    - Typical config fields: `dataset_folder`, `val_folder`, `test_folder`, `res_dir`, `epochs`, `batch_size`, `lr`, `npixel`, `device`, etc.
    - Output: Model checkpoint, logs, metrics, and plots in the results directory.

### Hyperparameter Tuning
- **Scripts:** `hyperparameter_tuning_s1.py`, `hyperparameter_tuning_s2.py`
- **How to Run:**
    ```bash
    python hyperparameter_tuning_s1.py --dataset_folder path/to/train --val_folder path/to/val --test_folder path/to/test --epochs 20 --res_dir hypertuning_results_s1
    python hyperparameter_tuning_s2.py --dataset_folder path/to/train --val_folder path/to/val --test_folder path/to/test --epochs 20 --res_dir hypertuning_results_s2
    ```
    - These scripts iterate over combinations of hyperparameters and log results in `hyperparameter_log.txt` and per-trial directories.
    - Use these before final training to identify optimal model settings.

### Inference
- **Scripts:** `inferencing/inference_s1.py`, `inferencing/inference_s2.py`
- **How to Run:**
    ```bash
    python inferencing/inference_s1.py --model_path path/to/model.pth.tar --data_dir path/to/infer_data --output_dir path/to/save_results --device cuda
    python inferencing/inference_s2.py --model_path path/to/model.pth.tar --data_dir path/to/infer_data --output_dir path/to/save_results --device cuda
    ```
    - `--model_path`: Trained model checkpoint
    - `--data_dir`: Folder with DATA and META subfolders (same as training format)
    - `--output_dir`: Where to save predictions/results
    - `--device`: 'cuda' or 'cpu'
    - Outputs: Predictions, confidence scores, metrics, and plots in the output directory.

---

## 2. Multi Sensor Workflow (sentinel_psetae/multi_sensor)

### Training
- **Script:** `train_fusion.py`
- **How to Run:**
    ```bash
    python train_fusion.py --config config.json
    ```
    - Expects a configuration file with paths for both S1 and S2 data, fusion model settings, and training parameters.
    - Output: Model checkpoint, logs, and results in the results directory.

### Inference
- **Scripts:**
    - `inference/inference_fusion_pse.py` (PixelSetEncoder fusion)
    - `inference/inference_fusion_softmax.py` (Softmax fusion)
    - `inference/inference_fusion_tae.py` (Temporal Attention Encoder fusion)
- **How to Run:**
    ```bash
    python inference/inference_fusion_pse.py --config config.json
    python inference/inference_fusion_softmax.py --config config.json
    python inference/inference_fusion_tae.py --config config.json
    ```
    - Each script expects a config file specifying paths to S1/S2 data, model checkpoint, and output settings.
    - Outputs: Fused predictions, metrics, and plots in the specified results directory.

---

## 3. When to Use Each Workflow
- **Single Sensor:**
    - Use when you want to train or evaluate models using only Sentinel-1 or only Sentinel-2 data.
    - Suitable for sensor-specific analysis, benchmarking, or ablation studies.
- **Multi Sensor (Fusion):**
    - Use when you want to leverage both Sentinel-1 and Sentinel-2 data for improved classification.
    - Recommended for final models or when maximizing accuracy is critical.
    - Choose the fusion script (`pse`, `softmax`, or `tae`) based on your preferred fusion strategy or ablation study.

---

## 4. General Notes
- Ensure your data folders follow the expected structure (DATA and META subfolders, .npy arrays, and JSON metadata).
- For hyperparameter tuning, review `hyperparameter_log.txt` and per-trial folders for best results.
- All scripts support CUDA for acceleration; set `--device cuda` for GPU usage.
- Outputs include model checkpoints, logs, metrics (accuracy, IoU, F1), and visualizations (confusion matrix, accuracy/loss curves).

---

For further details, consult the docstrings and comments in each script, or ask for specific usage examples.
