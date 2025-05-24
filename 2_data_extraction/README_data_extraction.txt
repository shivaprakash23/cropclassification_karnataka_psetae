# Data Extraction Scripts Overview

This directory contains scripts to extract satellite data from Google Earth Engine (GEE) for crop classification and related tasks. The main scripts are:

- `hls/hls_l30_extraction_final.py` – For HLS Landsat-8 (L30) data
- `hls/hls_s30_extraction_final.py` – For HLS Sentinel-2 (S30) data
- `sentinel/utils.py` – For Sentinel-1 and Sentinel-2 data (with additional utilities and advanced filtering)

## Common Workflow

1. **Initialization**:  
   All scripts initialize the Earth Engine API and authenticate with a specific project.

2. **Collection Filtering**:  
   - **Spatial Filter**: Data is filtered to the geometry of interest (e.g., farm parcels).
   - **Temporal Filter**: Data is filtered by user-specified start and end dates.
   - **Cloud Cover Filter**: For optical data, images are filtered by a maximum allowed cloud cover.

3. **Band Selection**:  
   - **HLS-L30 (Landsat-8)**:  
     Bands extracted: `['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']`
   - **HLS-S30 (Sentinel-2)**:  
     Bands extracted: `['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']`
   - **Sentinel-2 (via utils.py)**:  
     Bands extracted: `['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']`
   - **Sentinel-1 (via utils.py)**:  
     Bands extracted: `['VV', 'VH']` (dual-polarization SAR)

4. **Optional Band Calculations**:  
   - **NDVI**: For Sentinel-2, NDVI can be added as an extra band (`addNDVI` flag).

5. **Normalization**:  
   - All scripts perform min-max normalization using the 2nd and 98th percentiles for each band, computed per image.

6. **Additional Filtering/Processing**:
   - **Sentinel-1**:  
     - Filter by instrument mode, polarization, orbit pass, and optionally by relative orbit number.
     - Optional speckle filtering (temporal, mean, or median).
     - Reprojection to UTM Zone 43N (EPSG:32643) for Karnataka, India.
     - Clip to a buffered geometry to optimize speckle filtering.
   - **Monthly Reduction**:  
     - Optionally, the scripts can return a fixed number of images per month (e.g., the least cloudy).

7. **Output**:  
   - Extracted data and metadata are saved in NumPy arrays and JSON files, organized into `DATA` and `META` directories.
   - The scripts take a GeoJSON file of parcels and extract time series for each parcel.

## Example Usage

For HLS-L30:
```bash
python hls_l30_extraction_final.py --rpg_file path/to/parcels.geojson --start_date 2024-09-01 --end_date 2025-03-31 --cloud_cover 20
```

For HLS-S30:
```bash
python hls_s30_extraction_final.py --rpg_file path/to/parcels.geojson --start_date 2024-09-01 --end_date 2025-03-31 --cloud_cover 100
```

For Sentinel-1/2 (see `utils.py` for function usage):
- Use `get_collection()` with appropriate parameters for S1 or S2.

## Filtering and Preprocessing Summary

- **Cloud Filtering**:  
  - L30: `CLOUD_COVERAGE` attribute  
  - S30/Sentinel-2: `CLOUDY_PIXEL_PERCENTAGE` attribute
- **Spatial Filtering**:  
  - All scripts use `.filterBounds(geometry)`
- **Temporal Filtering**:  
  - All scripts use `.filterDate(start_date, end_date)`
- **Speckle Filtering (SAR)**:  
  - Temporal, mean, or median filters can be applied for Sentinel-1.
- **Normalization**:  
  - All bands are normalized per image using 2nd and 98th percentiles.
- **NDVI Calculation**:  
  - Optional for Sentinel-2.

---

This document summarizes the data extraction workflow and band/processing choices in the provided scripts. For more details, refer to the source code and inline comments.
