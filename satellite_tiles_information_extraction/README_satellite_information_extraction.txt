# Satellite Tiles Information Extraction - Usage Guide

This folder contains scripts to extract and analyze satellite tile coverage information for different sensors using Google Earth Engine (GEE).

---

## Common Requirements
- Python environment with `earthengine-api`, `datetime`, `json`, and standard libraries.
- GEE account and project initialization (already present in scripts).
- Input geometry (usually as GeoJSON or coordinates).
- Date range for the analysis.

---

## Scripts Overview

### 1. hls_l30_coverage.py
- **Purpose:** Extracts and analyzes coverage information for HLS L30 (Harmonized Landsat 8 & 9) tiles.
- **Main Functions:**
  - `find_mgrs_tile(geometry)`: Finds the MGRS tile for the given geometry using Sentinel-2 data.
  - `get_hls_l30_info(geometry, start_date, end_date, cloud_cover_threshold=80)`: Retrieves and simulates coverage info based on an 8-day revisit cycle.
  - `analyze_coverage(geojson_path, start_date, end_date, output_file, cloud_cover_threshold)`: Main function to run analysis.
  - `write_coverage_to_file(features, output_file)`: Exports results.
- **When to Use:** For HLS L30 data availability (e.g., time-series, crop monitoring).

---

### 2. hls_s30_coverage.py
- **Purpose:** Extracts and analyzes coverage information for HLS S30 (Harmonized Sentinel-2) tiles.
- **Main Functions:**
  - `find_mgrs_tile(geometry)`: Finds the MGRS tile for the given geometry.
  - `get_hls_s30_info(geometry, start_date, end_date, cloud_cover_threshold=80)`: Retrieves and simulates coverage info based on a 5-day revisit cycle.
  - `analyze_coverage(geojson_path, start_date, end_date, output_file)`: Main function for area analysis.
  - `write_coverage_to_file(features, output_file)`: Exports results.
- **When to Use:** For Sentinel-2 based HLS coverage assessments.

---

### 3. sentinel1_coverage.py
- **Purpose:** Extracts and analyzes coverage information for Sentinel-1 SAR data.
- **Main Functions:**
  - `get_sentinel1_info(geometry, start_date, end_date, orbit_pass)`: Retrieves Sentinel-1 coverage, filtering by orbit pass (ASCENDING/DESCENDING).
  - `analyze_sentinel1_coverage(results)`: Analyzes the retrieved coverage.
  - `write_coverage_to_file(features, output_file)`: Exports results.
- **When to Use:** For radar data availability, all-weather observations, or SAR revisit checks.

---

### 4. sentinel2_coverage.py
- **Purpose:** Extracts and analyzes coverage information for Sentinel-2 optical data.
- **Main Functions:**
  - `get_sentinel2_info(geometry, start_date, end_date)`: Retrieves Sentinel-2 coverage for the given area and dates.
  - `analyze_sentinel2_coverage(results)`: Analyzes the coverage statistics.
  - `write_coverage_to_file(features, output_file)`: Exports results.
- **When to Use:** For detailed Sentinel-2 availability checks (e.g., NDVI, land cover, crop classification).

---

## General Usage Steps

1. **Prepare Input:**
   - Define your area of interest (AOI) as a geometry or GeoJSON file.
   - Specify the date range for analysis.

2. **Run the Script:**
   - Execute the script from the command line or within a Python environment.
   - Example:
     ```bash
     python hls_l30_coverage.py --geojson_path your_aoi.geojson --start_date 2023-06-01 --end_date 2023-09-30 --output_file output.json
     ```

3. **Output:**
   - Results are saved to the specified output file, containing coverage statistics per tile and per date.

4. **Customization:**
   - Adjust cloud cover thresholds or revisit cycles as needed in the script arguments.

---

For more details, review the docstrings and comments within each script.
