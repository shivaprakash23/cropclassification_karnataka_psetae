# Sentinel Data Extraction Script

`sentinel_extraction.py`

This script provides a solution for extracting Sentinel-1 (S1) and Sentinel-2 (S2) time-series data using Google Earth Engine (GEE). It processes data for specified parcels, normalizes it, and saves it locally in a standardized format.

## Features

- **Dual Sensor Support**: Extracts data from both Sentinel-1 (GRD) and Sentinel-2 (SR) collections.
- **Flexible Filtering**: Allows filtering by date range, cloud cover (S2), and footprint ID (MGRS tile for S2, relative orbit number for S1).
- **Temporal Aggregation**: Supports monthly compositing or selection of a specific number of images per month.
- **S1 Preprocessing**: Includes options for multi-temporal speckle filtering (temporal, mean, median) and reprojection for Sentinel-1 data.
- **S2 Derived Index**: Optionally computes and includes NDVI for Sentinel-2 data.
- **Standardized Normalization**: Normalizes pixel values using 2nd and 98th percentiles for each image.
- **Metadata Extraction**: Captures geometric features, acquisition dates, and labels for each parcel.
- **Command-Line Interface**: All parameters are configurable via CLI arguments.
- **Structured Output**: Saves data as NumPy arrays (`.npy`) and metadata as JSON files (`.json`), suitable for machine learning workflows.

---

### Workflow Block Diagram

```
┌───────────────────────────────┐
│      Start / Main             │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Parse Arguments (CLI Inputs) │
│  (rpg_file, output_dir, col_id, dates, etc.)
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Initialize GEE & Prepare     │
│  Output Directories (DATA, META)│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Load GeoJSON Parcel File     │
│  (Polygons, Labels, IDs)      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  For Each Parcel:             │
│  ───────────────────────────  │
│  1. Build GEE Geometry        │
│  2. Get Sentinel Collection   │
│     (S1/S2 specific filters:  │
│      cloud, speckle, NDVI)    │
│  3. Normalize Bands           │
│  4. Extract Time Series       │
│     (ReduceRegion to List)    │
│  5. Convert to NumPy Array    │
│     (TxCxN)                   │
│  6. Extract Dates (DOA)       │
│  7. Extract Labels            │
│  8. Extract Geom. Features    │
│  9. Save Parcel .npy Data     │
│ 10. Log Errors if Any         │
│     (ignored_parcels.json)    │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Save Global Metadata Files    │
│ (dates.json, labels.json,     │
│  geomfeat.json)               │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      End / Print Summary      │
└───────────────────────────────┘
```

---

## Prerequisites

- Google Earth Engine account with Python API access.
- Python 3.7+ (or a version compatible with the dependencies).
- Required Python packages:
  ```bash
  pip install numpy tqdm earthengine-api
  ```

## Installation

1.  Install the required Python packages:
    ```bash
    pip install numpy tqdm earthengine-api
    ```

2.  Authenticate with Earth Engine (this is a one-time setup per machine/environment):
    ```bash
    earthengine authenticate
    ```
    Follow the on-screen instructions to authorize GEE access.

3.  **Configure GEE Project ID**:
    Open the `sentinel_extraction.py` script and replace `'your-project-id'` with your actual Google Earth Engine project ID in the `main()` function:
    ```python
    # In sentinel_extraction.py, within the main() function:
ee.Initialize(project='your-project-id')  # <-- REPLACE THIS with your GEE project ID
    ```

## Usage

### Basic Command Structure

```bash
python sentinel_extraction.py <rpg_file> <output_dir> --col_id <GEE_COLLECTION_ID> [options]
```

### Arguments

#### Positional Arguments:
-   `rpg_file`: (Required) Path to the GeoJSON file containing parcel polygons and attributes.
-   `output_dir`: (Required) Path to the directory where the output data will be saved.

#### Optional Arguments:
-   `--id_field TEXT`: Field name in the GeoJSON `properties` that contains the unique parcel identifier (default: `ID_PARCEL`).
-   `--label_names LIST`: List of field names in the GeoJSON `properties` to be used as labels (default: `CODE_GROUP`). Pass as space-separated values if multiple, e.g., `--label_names CODE_GROUP CROP_TYPE`.
-   `--col_id TEXT`: Google Earth Engine collection ID (default: `COPERNICUS/S2_SR`).
    -   For Sentinel-2: `COPERNICUS/S2_SR` or `COPERNICUS/S2_SR_HARMONIZED`.
    -   For Sentinel-1: `COPERNICUS/S1_GRD`.
-   `--start_date TEXT`: Start date for image collection filtering (format: YYYY-MM-DD, default: `2024-09-01`).
-   `--end_date TEXT`: End date for image collection filtering (format: YYYY-MM-DD, default: `2025-03-31`).
-   `--num_per_month INTEGER`: Number of scenes to select per month. If `0`, all available scenes passing other filters will be used (default: `0`).
-   `--footprint_id LIST`: List of granule/orbit identifiers to filter the collection. Pass as space-separated values.
    -   For Sentinel-1 (relative orbit numbers): e.g., `--footprint_id 153` or `--footprint_id 153 154`.
    -   For Sentinel-2 (MGRS tiles): e.g., `--footprint_id 30UUU` or `--footprint_id 30UUU 30UVU`.
    (Default: `None` - no footprint filtering).

#### Sentinel-1 Specific Arguments:
-   `--speckle_filter TEXT`: Type of speckle filter to apply to Sentinel-1 data. Options: `temporal`, `mean`, `median` (default: `temporal`).
-   `--kernel_size INTEGER`: Kernel/window size in pixels for focal speckle filters (`mean`, `median`) (default: `5`).

#### Sentinel-2 Specific Arguments:
-   `--cloud_cover INTEGER`: Maximum cloud cover percentage allowed for Sentinel-2 images (default: `80`).
-   `--addNDVI BOOLEAN`: If `True`, computes and appends NDVI as an additional band for Sentinel-2 images (default: `False`).

## Output Structure

The script creates the specified `output_dir` with the following subdirectories and files:

```
<output_dir>/
├── DATA/
│   ├── <parcel_id_1>.npy
│   ├── <parcel_id_2>.npy
│   └── ...
└── META/
    ├── dates.json
    ├── geomfeat.json
    ├── labels.json
    └── (ignored_parcels.json)  # Created if any parcels are skipped due to errors
```

### Data Format

1.  **NumPy Arrays (`DATA/<parcel_id>.npy`)**:
    -   Each file corresponds to a single parcel.
    -   The array shape is `(T, C, N)`:
        -   `T`: Time dimension (number of images/timestamps in the series for that parcel).
        -   `C`: Channel/band dimension (e.g., VV, VH for S1; B2-B12, NDVI for S2).
        -   `N`: Number of pixels within the parcel (flattened array of pixel values for each band at each timestamp).
    -   Pixel values are normalized using a 2nd to 98th percentile min-max scaling, applied per image.

2.  **Metadata Files (`META/*.json`)**:
    -   `dates.json`: A JSON object mapping each parcel ID (as a string) to a list of acquisition dates (format: `YYYYMMDD`) for its time series.
    -   `labels.json`: A JSON object mapping each parcel ID (as a string) to its corresponding label value(s) extracted from the GeoJSON.
    -   `geomfeat.json`: A JSON object mapping each parcel ID (as a string) to a list containing its perimeter (currently `[perimeter_value]`).
    -   `ignored_parcels.json` (optional): A JSON file where each line is an integer parcel ID that was skipped due to an error during processing.

## Error Handling

-   The script attempts to process each parcel independently.
-   If an error occurs during the processing of a specific parcel (e.g., no images found, GEE computation error), that parcel's ID is logged to `META/ignored_parcels.json`, and the script continues to the next parcel.
-   Basic error messages are printed to the console.

## Examples

### Example 1: Extract Sentinel-2 Data

```bash
python sentinel_extraction.py input_parcels.geojson sentinel2_output \
    --col_id "COPERNICUS/S2_SR_HARMONIZED" \
    --start_date "2023-04-01" \
    --end_date "2023-09-30" \
    --cloud_cover 15 \
    --num_per_month 1 \
    --addNDVI True \
    --footprint_id 43QEU 43QFV
```

### Example 2: Extract Sentinel-1 Data

```bash
python sentinel_extraction.py input_parcels.geojson sentinel1_output \
    --col_id "COPERNICUS/S1_GRD" \
    --start_date "2023-04-01" \
    --end_date "2023-09-30" \
    --num_per_month 0 \
    --speckle_filter "temporal" \
    --kernel_size 3 \
    --footprint_id 63 78
```

## Troubleshooting

1.  **Authentication Issues**:
    -   Ensure you have run `earthengine authenticate` and successfully logged in.
    -   Verify that your GEE account is active and has access to the required collections.
    -   Confirm the GEE Project ID in `sentinel_extraction.py` is correct and active.

2.  **Memory or Timeout Issues with GEE**:
    -   If processing a very large number of parcels or very large individual parcels, GEE might time out or run into computation limits.
    -   Try reducing the date range or processing a smaller subset of parcels first.
    -   Simplify geometries if they are overly complex, though the script uses `reduceRegion` which is generally efficient.

3.  **No Data Extracted / All Parcels Ignored**:
    -   Double-check the `start_date`, `end_date`, and `footprint_id` parameters to ensure they cover the area and time of interest for your parcels and the chosen `col_id`.
    -   For Sentinel-2, `cloud_cover` might be too restrictive.
    -   Verify that the parcel coordinates in your GeoJSON are correct (e.g., WGS84).

4.  **`argparse` List Input**:
    -   For arguments like `--footprint_id` or `--label_names` that expect a list, provide values separated by spaces (e.g., `--footprint_id tile1 tile2`).
