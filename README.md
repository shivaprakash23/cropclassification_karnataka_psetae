# PSETAE Crop Classification & Satellite Data Pipeline

This repository provides a full pipeline for extracting satellite data (HLS, Sentinel-1/2), training and inferring crop classification models using the PSETAE architecture, and analyzing satellite tile coverage. The codebase is modular, with dedicated folders for data extraction, model training (HLS L30/S30, Sentinel), and tile information extraction.

---

## Directory Structure

```
psetae_github_publish/
├── data_extraction/                  # Scripts for extracting HLS & Sentinel data
│   ├── hls/                          # Unified HLS L30/S30 extraction
│   ├── sentinel/                     # Sentinel-1/2 extraction utilities
│   └── ...
├── hls_l30_psetae/                   # PSETAE for HLS L30 (Landsat)
├── hls_s30_psetae/                   # PSETAE for HLS S30 (Sentinel-2)
├── sentinel_psetae/                  # PSETAE for Sentinel-1/2 (single & multi-sensor)
├── satellite_tiles_information_extraction/  # Scripts for tile coverage analysis
└── ...
```

---

## 1. Data Extraction Pipeline

### HLS L30 & S30 Extraction
- Located in `data_extraction/hls/`
- Unified script for Harmonized Landsat Sentinel-2 (HLS) L30 (Landsat) and S30 (Sentinel-2)
- Handles cloud cover, band selection, normalization, and metadata
- Output: `.npy` arrays (DATA) and JSON metadata (META)
- See `data_extraction/hls/README.md` for usage and workflow diagram

### Sentinel-1/2 Extraction
- Located in `data_extraction/sentinel/`
- Utilities for extracting Sentinel-1 (SAR) and Sentinel-2 (optical) data
- Supports band selection, speckle filtering, NDVI, and monthly reduction
- Output: Structured data for PSETAE models

---

## 2. PSETAE for HLS L30 & S30

### HLS L30 PSETAE (`hls_l30_psetae/`)
- Train and infer crop classification using PSETAE on HLS L30 data
- Training: `train.py` with config or CLI arguments
- Outputs: Model checkpoints, logs, metrics, test results
- See `hls_l30_psetae/README_hls_l30.txt` for details

### HLS S30 PSETAE (`hls_s30_psetae/`)
- Train and infer with PSETAE on HLS S30 data
- Similar workflow to L30, with Sentinel-2 bands
- Outputs: Model checkpoints, logs, metrics, test results
- See `hls_s30_psetae/README.txt` for details

---

## 3. Sentinel PSETAE (`sentinel_psetae/`)

### Single Sensor
- Train and infer with Sentinel-1 or Sentinel-2 using PSETAE
- Hyperparameter tuning scripts for both sensors
- Inference scripts for each sensor

### Multi Sensor
- Train and infer with fused Sentinel-1 & Sentinel-2 data
- Dedicated training and inference scripts
- See `sentinel_psetae/README_sentinel.txt` for all workflows

---

## 4. Satellite Tiles Information Extraction

- Located in `satellite_tiles_information_extraction/`
- Scripts for analyzing tile coverage for HLS L30, HLS S30, Sentinel-1, and Sentinel-2
- Functions to find MGRS tiles, simulate revisit cycles, and export coverage info
- Useful for understanding data availability and planning analyses
- See `satellite_tiles_information_extraction/README_satellite_information_extraction.txt`

---

## 5. Example Workflow

1. **Extract Data:**
   - Use scripts in `data_extraction/hls/` or `data_extraction/sentinel/` to generate `.npy` and metadata files for your parcels
2. **Train Model:**
   - Use `train.py` in the relevant PSETAE folder (HLS L30, HLS S30, or Sentinel)
3. **Tune Hyperparameters:**
   - Run the provided tuning scripts to optimize model settings
4. **Inference:**
   - Use the inference scripts to generate predictions on new data
5. **Analyze Coverage:**
   - Use tile information scripts to assess data availability for your study area

---

## References & Credits

- Developed by Shivaprakash Yaragal, Lund University, Sweden
- For crop classification and time series extraction using multi-sensor satellite data in Karnataka, India
- See individual subfolder READMEs for detailed usage and script documentation
