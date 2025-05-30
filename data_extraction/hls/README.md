# HLS Data Extraction Script

This script provides a unified solution for extracting Harmonized Landsat Sentinel-2 (HLS) data using Google Earth Engine (GEE). It supports both L30 (Landsat) and S30 (Sentinel-2) data extraction with standardized output format and error handling.

## Features

- **Unified Processing**: Single script handling both L30 and S30 data extraction
- **Standardized Output**: Consistent data structure and metadata format
- **Robust Error Handling**: Comprehensive error logging and debugging information
- **Cloud Cover Control**: Default 20% cloud cover threshold for both sensors
- **Automatic Band Selection**:
  - L30: 8 bands (B2, B3, B4, B5, B6, B7, B10, B11)
  - S30: 10 bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)

---

### Workflow Block Diagram

```
┌────────────────────┐
│   Start / Main     │
└─────────┬──────────┘
          │
          ▼
┌───────────────────────────────┐
│ Parse Arguments (CLI Inputs)  │
└─────────┬─────────────────────┘
          │
          ▼
┌───────────────────────────────┐
│  Load GeoJSON Parcel File     │
└─────────┬─────────────────────┘
          │
          ▼
┌───────────────────────────────┐
│  For Each Parcel:             │
│  ───────────────────────────  │
│  1. Build GEE Geometry        │
│  2. Get HLS Collection        │
│     (L30/S30, cloud, bands)   │
│  3. Normalize Bands           │
│  4. Extract Time Series       │
│  5. Pad/Shape Data (TxCxN)    │
│  6. Extract Dates             │
│  7. Extract Labels            │
│  8. Extract Geom. Features    │
│  9. Save Data & Meta          │
│ 10. Log Errors if Any         │
└─────────┬─────────────────────┘
          │
          ▼
┌───────────────────────────────┐
│ Save Global Metadata Files    │
│ (dates.json, labels.json,     │
│  geomfeat.json, logs, etc.)   │
└─────────┬─────────────────────┘
          │
          ▼
┌────────────────────┐
│      End           │
└────────────────────┘
```

---

## Prerequisites

- Google Earth Engine account with Python API access
- Python 3.7+
- Required packages:
  ```bash
  pip install numpy tqdm earthengine-api
  ```

## Installation

1. Install required packages:
   ```bash
   pip install numpy tqdm earthengine-api
   ```

2. Authenticate with Earth Engine (one-time setup):
   ```bash
   earthengine authenticate
   ```

## Usage

### Basic Command

```bash
python hls_extraction.py --rpg_file <path_to_geojson> --sensor_type <L30 or S30>
```

### Full Options

```bash
python hls_extraction.py \
    --rpg_file path/to/parcels.geojson \
    --label_names CODE_GROUP OTHER_LABEL \
    --id_field ID_PARCEL \
    --output_dir hls_extraction_numpy_files \
    --start_date 2024-09-01 \
    --end_date 2025-03-31 \
    --cloud_cover 20 \
    --sensor_type L30
```

### Arguments

- `--rpg_file`: Path to GeoJSON file containing parcel polygons (required)
- `--label_names`: List of label fields to extract (default: `CODE_GROUP`)
- `--id_field`: Field name for parcel IDs (default: `ID_PARCEL`)
- `--output_dir`: Base output directory (default: `hls_extraction_numpy_files`)
- `--start_date`: Start date for extraction (default: `2024-09-01`)
- `--end_date`: End date for extraction (default: `2025-03-31`)
- `--cloud_cover`: Maximum cloud cover percentage (default: 20)
- `--sensor_type`: Type of sensor data to extract (`L30` or `S30`, default: `L30`)

## Output Structure

The script creates separate output directories for L30 and S30 data:
```
hls_extraction_numpy_files_l30/
├── DATA/
│   ├── parcel1.npy  # TxCxN array
│   ├── parcel2.npy
│   └── ...
└── META/
    ├── dates.json       # Temporal information
    ├── labels.json      # Parcel labels
    ├── geomfeat.json    # Geometric features
    ├── ignored_parcels.json  # Failed parcels
    └── debug_log.txt    # Detailed error logs

hls_extraction_numpy_files_s30/
├── DATA/
└── META/
    └── ...
```

### Data Format

1. **Numpy Arrays** (`DATA/*.npy`):
   - Shape: `TxCxN`
     - T: Time dimension (number of timestamps)
     - C: Channel dimension (8 bands for L30, 10 bands for S30)
     - N: Number of pixels dimension
   - Normalized using min-max scaling (2nd to 98th percentile)

2. **Metadata Files**:
   - `dates.json`: Timestamp information for each parcel
   - `labels.json`: Extracted label values (converts to -1 for null/invalid)
   - `geomfeat.json`: Geometric features (perimeter, shape ratio)
   - `ignored_parcels.json`: List of failed parcels
   - `debug_log.txt`: Detailed error logs and debugging information

## Error Handling

The script includes comprehensive error handling:
- Invalid/null labels are converted to -1
- Failed parcels are logged in `ignored_parcels.json`
- Detailed error information is saved in `debug_log.txt`
- Inhomogeneous array shapes are automatically handled with padding

## Examples

### Extract L30 Data
```bash
python hls_extraction.py \
    --rpg_file parcels.geojson \
    --sensor_type L30 \
    --output_dir hls_data
```

### Extract S30 Data
```bash
python hls_extraction.py \
    --rpg_file parcels.geojson \
    --sensor_type S30 \
    --output_dir hls_data
```

### Extract Both L30 and S30
```bash
# Extract L30 data
python hls_extraction.py --rpg_file parcels.geojson --sensor_type L30 --output_dir hls_data

# Extract S30 data
python hls_extraction.py --rpg_file parcels.geojson --sensor_type S30 --output_dir hls_data
```

## Troubleshooting

1. **Authentication Issues**
   - Ensure you've run `earthengine authenticate`
   - Check if your GEE account is active

2. **Memory Issues**
   - Process fewer parcels at a time
   - Reduce the date range
   - Increase cloud cover threshold

3. **No Data Found**
   - Verify parcel coordinates
   - Check date range
   - Adjust cloud cover threshold

4. **Label Errors**
   - Check `debug_log.txt` for details
   - Verify label format in GeoJSON
   - Invalid labels are set to -1

## Credits
Developed by Shivaprakash Yaragal, Lund University, Sweden
Developed for crop classification and time series extraction using HLS data in Karnataka, India.
