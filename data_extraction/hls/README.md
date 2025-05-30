# HLS Data Extraction Script

This script provides a unified solution for extracting Harmonized Landsat Sentinel-2 (HLS) data using Google Earth Engine (GEE). It supports both L30 (Landsat-based) and S30 (Sentinel-2-based) data extraction in a single script.

## Features

- Unified extraction for both L30 and S30 data
- Automatic band selection based on sensor type
- Min-max normalization (2nd to 98th percentile)
- Cloud cover filtering
- Geometric feature extraction
- Structured output format

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
- `--cloud_cover`: Maximum cloud cover percentage
  - Default for L30: 20%
  - Default for S30: 100%
- `--sensor_type`: Type of sensor data to extract (`L30` or `S30`, default: `L30`)

## Output Structure

The script creates separate output directories for L30 and S30 data:
```
hls_extraction_numpy_files_l30/
├── DATA/
│   ├── parcel1.npy
│   ├── parcel2.npy
│   └── ...
└── META/
    ├── parcel1.npy
    ├── parcel2.npy
    └── ...

hls_extraction_numpy_files_s30/
├── DATA/
│   └── ...
└── META/
    └── ...
```

### Data Format

1. DATA files (`DATA/*.npy`):
   - NumPy arrays containing:
     - Time series dates
     - Normalized band values

2. META files (`META/*.npy`):
   - Geometric features:
     - Perimeter
     - Perimeter/area ratio
     - Bounding box
   - Label values

## Examples

### Extract L30 Data
```bash
python hls_extraction.py \
    --rpg_file parcels.geojson \
    --sensor_type L30 \
    --cloud_cover 20
```

### Extract S30 Data
```bash
python hls_extraction.py \
    --rpg_file parcels.geojson \
    --sensor_type S30 \
    --cloud_cover 100
```

### Extract Both L30 and S30
```bash
# Extract L30 data
python hls_extraction.py --rpg_file parcels.geojson --sensor_type L30

# Extract S30 data
python hls_extraction.py --rpg_file parcels.geojson --sensor_type S30
```

## Sensor Differences

### L30 (Landsat)
- Collection: `NASA/HLS/HLSL30/v002`
- Bands: `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B10`, `B11`
- Default cloud cover: 20%

### S30 (Sentinel-2)
- Collection: `NASA/HLS/HLSS30/v002`
- Bands: `B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`
- Default cloud cover: 100%

## Troubleshooting

1. **Authentication Issues**
   - Ensure you've run `earthengine authenticate`
   - Check if your GEE account is active

2. **Memory Issues**
   - Process fewer parcels at a time
   - Reduce the date range
   - Increase cloud cover threshold

3. **No Data Found**
   - Check if parcels are within the satellite coverage area
   - Verify date range
   - Adjust cloud cover threshold

## Credits

Developed for crop classification and time series extraction using HLS data in Karnataka, India.

## License

This project is licensed under the MIT License.
