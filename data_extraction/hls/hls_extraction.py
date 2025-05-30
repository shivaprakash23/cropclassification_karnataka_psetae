#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HLS (L30 & S30) Extraction Script
This script extracts Harmonized Landsat Sentinel-2 (HLS) data using Google Earth Engine.
Supports both L30 (Landsat) and S30 (Sentinel-2) data extraction.
"""

import os
import json
import argparse
import numpy as np
import ee
from datetime import datetime
import warnings
from tqdm import tqdm

# Initialize Earth Engine with project
ee.Initialize(project='ee-shivaprakashssy-psetae-ka28')

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
    collection_id = "NASA/HLS/HLS" + sensor_type + "/v002"
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
    else:  # S30
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    
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
                   end_date='2025-03-31', cloud_cover=None, sensor_type='L30'):
    """Main function to prepare dataset from GeoJSON file
    
    Args:
        rpg_file: Path to GeoJSON file containing parcel polygons
        label_names: List of label fields to extract
        id_field: Field name for parcel IDs
        output_dir: Output directory for extracted data
        start_date: Start date for extraction (YYYY-MM-DD)
        end_date: End date for extraction (YYYY-MM-DD)
        cloud_cover: Maximum allowed cloud cover percentage (default: 20 for L30, 100 for S30)
        sensor_type: Type of sensor data to extract ('L30' or 'S30')
    """
    # Set default cloud cover based on sensor type
    if cloud_cover is None:
        cloud_cover = 20 if sensor_type == 'L30' else 100
        
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
    
    print(f"Processing {len(polygons)} parcels")
    
    # process each parcel
    for parcel_id, polygon in tqdm(polygons.items()):
        try:
            # create geometry
            coords = [[[x[0], x[1]] for x in polygon]]
            geometry = ee.Geometry.Polygon(coords)
            
            # get collection
            collection = get_collection(geometry, start_date, end_date, cloud_cover, sensor_type)
            size = collection.size().getInfo()
            
            if size == 0:
                print(f"No images found for parcel {parcel_id}")
                continue
                
            # get time series
            def get_values(img):
                values = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=30
                )
                return ee.Feature(None, {
                    'values': values,
                    'date': img.get('system:time_start')
                })
            
            time_series = collection.map(get_values).getInfo()
            
            # extract values and dates
            dates = []
            values = []
            
            for feat in time_series['features']:
                props = feat['properties']
                if all(v is not None for v in props['values'].values()):
                    dates.append(props['date'])
                    values.append([float(v) for v in props['values'].values()])
            
            if len(dates) == 0:
                print(f"No valid data for parcel {parcel_id}")
                continue
                
            # compute geometric features
            perimeter, perimeter_area_ratio, bbox = geom_features(geometry)
            
            # save data
            np.save(os.path.join(output_dir, 'DATA', f"{parcel_id}.npy"),
                   np.array([dates, values], dtype=object))
            
            # save metadata
            meta = {
                'perimeter': perimeter,
                'perimeter_area_ratio': perimeter_area_ratio,
                'bbox': bbox
            }
            for l in label_names:
                meta[l] = lab_rpg[l][parcel_id]
            
            np.save(os.path.join(output_dir, 'META', f"{parcel_id}.npy"), meta)
            
        except Exception as e:
            print(f"Error processing parcel {parcel_id}: {str(e)}")
            continue
    
    print(f"Processing completed in {datetime.now() - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract HLS (L30/S30) data for parcels')
    parser.add_argument('--rpg_file', type=str, required=True,
                      help='Path to GeoJSON file')
    parser.add_argument('--label_names', nargs='+', default=['CODE_GROUP'],
                      help='Label names to extract')
    parser.add_argument('--id_field', type=str, default='ID_PARCEL',
                      help='Field name for parcel IDs')
    parser.add_argument('--output_dir', type=str, default='hls_extraction_numpy_files',
                      help='Output directory')
    parser.add_argument('--start_date', type=str, default='2024-09-01',
                      help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-03-31',
                      help='End date (YYYY-MM-DD)')
    parser.add_argument('--cloud_cover', type=float,
                      help='Maximum cloud cover percentage (default: 20 for L30, 100 for S30)')
    parser.add_argument('--sensor_type', type=str, choices=['L30', 'S30'], default='L30',
                      help='Type of sensor data to extract (L30 or S30)')
    
    args = parser.parse_args()
    prepare_dataset(args.rpg_file, args.label_names, args.id_field,
                   args.output_dir, args.start_date, args.end_date,
                   args.cloud_cover, args.sensor_type)
