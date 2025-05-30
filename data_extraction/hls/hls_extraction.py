#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HLS (L30 & S30) Extraction Script
This script extracts Harmonized Landsat Sentinel-2 (HLS) data using Google Earth Engine.
Supports both L30 (Landsat) and S30 (Sentinel-2) data extraction with standardized output format.
"""

import os
import json
import argparse
import numpy as np
import ee
from datetime import datetime
import warnings
from tqdm import tqdm
import traceback

# Initialize Earth Engine with project
ee.Initialize(project='your project')

def get_collection(geometry, start_date, end_date, cloud_cover, sensor_type='L30'):
    """Get HLS collection filtered by date, bounds, and cloud cover
    
    Args:
        geometry: GEE geometry object
        start_date: Start date for filtering
        end_date: End date for filtering
        cloud_cover: Maximum allowed cloud cover percentage
        sensor_type: Type of sensor data to extract ('L30' or 'S30')
    """
    print(f"Fetching {sensor_type} collection for dates: {start_date} to {end_date}")
    
    # Initialize collection based on sensor type
    collection_id = f"NASA/HLS/HLS{sensor_type}/v002"
    collection = ee.ImageCollection(collection_id)
    print(f"Initial collection accessed: {sensor_type}")
    
    # Filter by date
    collection = collection.filterDate(start_date, end_date)
    
    # Filter by bounds
    collection = collection.filterBounds(geometry)
    
    # Filter by cloud cover
    collection = collection.filter(ee.Filter.lte('CLOUD_COVERAGE', cloud_cover))
    print(f"Filtering {sensor_type} collection for cloud cover: {cloud_cover}")
    
    # Select required bands based on sensor type
    if sensor_type == 'L30':
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
    elif sensor_type == 'S30':
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    else:
        raise ValueError("Invalid sensor type. Must be 'L30' or 'S30'")
    
    collection = collection.select(bands)
    
    # Get normalization statistics
    collection = collection.map(lambda img: img.set('stats', 
        ee.Image(img).reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]), 
            bestEffort=True
        )
    ))
    
    # Apply normalization
    collection = collection.map(normalize)
    
    return collection

def normalize(img):
    """Min-max normalization using 2 & 98 percentile"""
    try:
        stats = ee.Dictionary(img.get('stats'))
        bandNames = img.bandNames()
        
        def _normalize(bandName):
            bandName = ee.String(bandName)
            min_key = ee.String(bandName).cat('_p2')
            max_key = ee.String(bandName).cat('_p98')
            min_value = ee.Number(stats.get(min_key))
            max_value = ee.Number(stats.get(max_key))
            
            return img.select(bandName).subtract(min_value).divide(max_value.subtract(min_value))
        
        normalized = ee.ImageCollection(bandNames.map(_normalize)).toBands().rename(bandNames)
        return normalized.copyProperties(img, ["system:time_start", "stats"])
    except Exception as e:
        print(f"Error in normalize function: {str(e)}")
        return img
        
def geom_features(geometry):
    """Compute geometric features for a parcel"""
    # computes geometric info per parcel
    area = geometry.area().getInfo()
    perimeter = geometry.perimeter().getInfo()
    bbox = geometry.bounds().getInfo()
    return perimeter, perimeter/area, bbox

def prepare_dataset(rpg_file, label_names=['CODE_GROUP'], id_field='ID_PARCEL', 
                   output_dir='hls_extraction_numpy_files', start_date='2024-09-01', 
                   end_date='2025-03-31', cloud_cover=20, sensor_type='L30'):
    """Main function to prepare dataset from GeoJSON file
    
    Args:
        rpg_file: Path to GeoJSON file containing parcel polygons
        label_names: List of label fields to extract
        id_field: Field name for parcel IDs
        output_dir: Output directory for extracted data
        start_date: Start date for extraction (YYYY-MM-DD)
        end_date: End date for extraction (YYYY-MM-DD)
        cloud_cover: Maximum allowed cloud cover percentage (default: 20)
        sensor_type: Type of sensor data to extract ('L30' or 'S30')
    """
    warnings.filterwarnings('error', category=DeprecationWarning)
    start = datetime.now()

    # prepare output directories
    output_dir = f"{output_dir}_{sensor_type.lower()}"
    os.makedirs(os.path.join(output_dir, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'META'), exist_ok=True)
    
    # load GeoJSON
    with open(rpg_file) as f:
        data = json.load(f)
    
    # extract polygons and labels
    polygons = {}
    lab_rpg = {l:{} for l in label_names}
    
    for feature in data['features']:
        parcel_id = feature['properties'][id_field]
        polygons[parcel_id] = feature['geometry']['coordinates'][0]
        for l in label_names:
            lab_rpg[l][parcel_id] = feature['properties'][l]
    
    # dict of global metadata to store parcel dates/labels
    dates = {k:[] for k in list(polygons.keys())}
    labels = dict([(l, {}) for l in lab_rpg.keys()])
    # For geometric features
    geom_feats = {k: {} for k in list(polygons.keys())}

    # counter for ignored parcels
    ignored = 0

    # iterate parcels
    for parcel_id, polygon in tqdm(polygons.items()):
        print(f"\nProcessing parcel {parcel_id}...")
        try:
            # Convert to GEE geometry
            geometry = ee.Geometry.Polygon(polygon)
            print("Geometry converted to GEE format")

            # Get collection with validation
            collection = get_collection(geometry, start_date, end_date, cloud_cover, sensor_type)
            print("Collection retrieved")

            # Get time series array with all bands
            collection = collection.map(lambda img: img.set('temporal', 
                ee.Image(img).reduceRegion(
                    reducer=ee.Reducer.toList(), 
                    geometry=geometry, 
                    scale=30,
                    maxPixels=1e9
                ).values()
            ))
            print("Time series data extracted")

            # Query pre-selected collection & make numpy array - only do this once
            temporal_array = collection.aggregate_array('temporal').getInfo()
            if not temporal_array:
                raise ValueError("No temporal data retrieved")

            print(f"Temporal data retrieved: {len(temporal_array)} timestamps")
            
            # Convert to numpy array
            try:
                np_all_dates = np.array(temporal_array)
                print(f"Array shape: {np_all_dates.shape}")
                assert np_all_dates.shape[-1] > 0, "No valid pixels found"
            except ValueError as e:
                if "inhomogeneous shape" in str(e):
                    # This happens when pixels have different numbers of values
                    # We need to handle this case by padding or truncating
                    print("Handling inhomogeneous array shape...")
                    
                    # Find the maximum number of pixels in any band
                    max_pixels = 0
                    for t in temporal_array:
                        for band_values in t:
                            if isinstance(band_values, list):
                                max_pixels = max(max_pixels, len(band_values))
                    
                    # Create a padded array with consistent dimensions
                    T = len(temporal_array)  # Time dimension
                    C = len(temporal_array[0]) if T > 0 else 0  # Channel dimension
                    N = max_pixels  # Number of pixels dimension
                    
                    print(f"Creating padded array with dimensions: {T}x{C}x{N}")
                    padded_array = np.zeros((T, C, N))
                    
                    # Fill the array with available values
                    for t in range(T):
                        for c in range(C):
                            if t < len(temporal_array) and c < len(temporal_array[t]):
                                values = temporal_array[t][c]
                                if isinstance(values, list):
                                    n_values = len(values)
                                    padded_array[t, c, :n_values] = values
                    
                    np_all_dates = padded_array
                    print(f"Padded array shape: {np_all_dates.shape}")
                else:
                    raise

            # create date metadata
            date_series = collection.aggregate_array('system:time_start').getInfo()
            dates[str(parcel_id)] = [datetime.fromtimestamp(d/1000).strftime('%Y-%m-%d') for d in date_series]

            # save labels with error handling
            for l in labels.keys():
                try:
                    if lab_rpg[l][parcel_id] is None:
                        labels[l][parcel_id] = -1
                    else:
                        labels[l][parcel_id] = int(lab_rpg[l][parcel_id])
                except (ValueError, TypeError) as e:
                    error_msg = f"Label conversion error for parcel {parcel_id}, label {l}: {str(e)}"
                    print(error_msg)
                    with open(os.path.join(output_dir, 'META', 'debug_log.txt'), 'a+') as file:
                        file.write(f"\n{error_msg}\n")
                    labels[l][parcel_id] = -1
            
            # Calculate and save geometric features
            perimeter, shape_ratio, bbox = geom_features(geometry)
            geom_feats[str(parcel_id)] = [int(perimeter)]
            print(f"Geometric features calculated for parcel {parcel_id}")
            
            # save .npy 
            save_path = os.path.join(output_dir, 'DATA', str(parcel_id))
            np.save(save_path, np_all_dates)
            print(f"Saved numpy file to: {save_path}.npy")
            
        except Exception as e:
            print(f'Error in parcel {parcel_id}: {str(e)}')
            
            # Log to ignored_parcels.json
            with open(os.path.join(output_dir, 'META', 'ignored_parcels.json'), 'a+') as file:
                file.write(json.dumps(int(parcel_id))+'\n')
            
            # Detailed error logging
            with open(os.path.join(output_dir, 'META', 'debug_log.txt'), 'a+') as file:
                file.write(f"\nError in parcel {parcel_id} at {datetime.now()}:\n")
                file.write(f"{str(e)}\n")
                file.write(traceback.format_exc())
            
            ignored += 1

    # save global metadata (parcel dates, labels, and geometric features)
    with open(os.path.join(output_dir, 'META', 'geomfeat.json'), 'w') as file:
        file.write(json.dumps(geom_feats, indent=4))
        
    with open(os.path.join(output_dir, 'META', 'labels.json'), 'w') as file:
        file.write(json.dumps(labels, indent=4))
        
    with open(os.path.join(output_dir, 'META', 'dates.json'), 'w') as file:
        file.write(json.dumps(dates, indent=4))
        
    # print stats
    end = datetime.now()
    print('Extraction completed in {} seconds'.format((end - start).seconds))
    print('Ignored {} parcels out of {}'.format(ignored, len(polygons)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract HLS (L30/S30) data for parcels')
    parser.add_argument('--rpg_file', type=str, required=True, help='Path to GeoJSON file')
    parser.add_argument('--label_names', nargs='+', default=['CODE_GROUP'], help='Label names to extract')
    parser.add_argument('--id_field', type=str, default='ID_PARCEL', help='Field name for parcel ID')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--start_date', type=str, default='2024-09-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-03-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--cloud_cover', type=float, default=20, help='Maximum cloud cover percentage')
    parser.add_argument('--sensor_type', type=str, choices=['L30', 'S30'], default='L30',
                      help='Type of sensor data to extract (L30 or S30)')
    
    args = parser.parse_args()
    
    prepare_dataset(
        rpg_file=args.rpg_file,
        label_names=args.label_names,
        id_field=args.id_field,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        cloud_cover=args.cloud_cover,
        sensor_type=args.sensor_type
    )
