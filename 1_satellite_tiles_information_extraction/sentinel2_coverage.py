import ee
import os
from datetime import datetime, timedelta
import json
from collections import defaultdict

# Initialize Earth Engine with your project
ee.Initialize(project='ee-shivaprakashssy-psetae')

def get_sentinel2_info(geometry, start_date, end_date):
    """Get Sentinel-2 coverage information"""
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date)
    
    def get_metadata(image):
        return ee.Feature(None, {
            'mgrs_tile': image.get('MGRS_TILE'),
            'granule_id': image.get('system:index'),
            'date': image.date().format('YYYY-MM-dd'),
            'cloud_cover': image.get('CLOUDY_PIXEL_PERCENTAGE'),
            'platform': image.get('SPACECRAFT_NAME'),
            'orbit_direction': image.get('SENSING_ORBIT_DIRECTION')
        })
    
    metadata = collection.map(get_metadata)
    return metadata.getInfo()

def analyze_sentinel2_coverage(results):
    """Analyze Sentinel-2 coverage statistics"""
    track_data = defaultdict(lambda: {'dates': [], 'platforms': set()})
    if results is None or 'features' not in results:
        print("Warning: No Sentinel-2 data found for the given parameters")
        return track_data

    for feature in results['features']:
        props = feature['properties']
        date = props.get('date')
        platform = props.get('platform')
        orbit_direction = props.get('orbit_direction')
        cloud_cover = props.get('cloud_cover')
        mgrs_tile = props.get('mgrs_tile')

        if date and mgrs_tile:
            track_data[mgrs_tile]['dates'].append(date)
            if platform:
                track_data[mgrs_tile]['platforms'].add(platform)
            if orbit_direction:
                track_data[mgrs_tile]['orbit_direction'] = orbit_direction
            if cloud_cover is not None:
                track_data[mgrs_tile]['cloud_cover'] = cloud_cover

    return track_data

def write_coverage_to_file(features, output_file):
    """Write coverage information to a file"""
    # Group features by MGRS tile
    tiles = defaultdict(list)
    for feature in features:
        tiles[feature['properties']['mgrs_tile']].append(feature)
    
    with open(output_file, 'w') as f:
        f.write("Sentinel-2 Coverage Analysis\n")
        f.write("======================\n\n")
        
        f.write(f"Found data for {len(tiles)} MGRS tiles\n")
        for tile, tile_features in tiles.items():
            f.write(f"MGRS Tile {tile}: {len(tile_features)} acquisitions\n")
        f.write("\n")
        
        for tile, features in tiles.items():
            # Calculate monthly statistics
            monthly_stats = defaultdict(lambda: {'count': 0, 'avg_cloud': 0.0})
            yearly_stats = defaultdict(lambda: {'count': 0, 'avg_cloud': 0.0})
            
            for feature in features:
                date_obj = datetime.strptime(feature['properties']['date'], '%Y-%m-%d')
                month_key = (date_obj.year, date_obj.month)
                year_key = date_obj.year
                
                # Update monthly stats
                monthly_stats[month_key]['count'] += 1
                monthly_stats[month_key]['avg_cloud'] += feature['properties']['cloud_cover']
                
                # Update yearly stats
                yearly_stats[year_key]['count'] += 1
                yearly_stats[year_key]['avg_cloud'] += feature['properties']['cloud_cover']
            
            # Calculate averages
            for stats in monthly_stats.values():
                stats['avg_cloud'] /= stats['count']
            for stats in yearly_stats.values():
                stats['avg_cloud'] /= stats['count']
            
            f.write(f"Detailed Coverage for MGRS Tile {tile}\n")
            f.write("-" * 40 + "\n")
            f.write("Year | Month | Date       | Platform  | Cloud Cover\n")
            f.write("-" * 55 + "\n")
            for feature in sorted(features, key=lambda x: x['properties']['date']):
                date_obj = datetime.strptime(feature['properties']['date'], '%Y-%m-%d')
                year = date_obj.year
                month = date_obj.strftime('%b')
                f.write(f"{year} | {month:>5} | {feature['properties']['date']} | {feature['properties']['platform']:<9} | {feature['properties']['cloud_cover']:.1f}%\n")
            f.write("\n")
            
            # Write monthly statistics
            f.write("Monthly Statistics\n")
            f.write("-" * 40 + "\n")
            f.write("Year | Month | Acquisitions | Avg Cloud Cover\n")
            f.write("-" * 45 + "\n")
            for (year, month), stats in sorted(monthly_stats.items()):
                month_name = datetime(year, month, 1).strftime('%b')
                f.write(f"{year} | {month_name:>5} | {stats['count']:>12} | {stats['avg_cloud']:>13.1f}%\n")
            f.write("\n")
            
            # Write yearly statistics
            f.write("Yearly Statistics\n")
            f.write("-" * 40 + "\n")
            f.write("Year | Acquisitions | Avg Cloud Cover\n")
            f.write("-" * 40 + "\n")
            for year, stats in sorted(yearly_stats.items()):
                f.write(f"{year} | {stats['count']:>12} | {stats['avg_cloud']:>13.1f}%\n")
            f.write("\n")
        
        # Write footer with time range
        all_dates = [datetime.strptime(f['properties']['date'], '%Y-%m-%d') for f in features]
        if all_dates:
            start = min(all_dates).strftime('%Y-%m-%d')
            end = max(all_dates).strftime('%Y-%m-%d')
            f.write(f"Time Range: {start} to {end}\n")
            f.write(f"Total Acquisitions: {len(features)}\n")
    
    print(f"\nCoverage information saved to: {os.path.abspath(output_file)}")

def analyze_coverage(geojson_path, start_date=None, end_date=None, output_file=None):
    """
    Analyze Sentinel-2 coverage for a given GeoJSON area
    
    Args:
        geojson_path (str): Path to the GeoJSON file
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        output_file (str, optional): Path to save the coverage information
    """
    try:
        # Load GeoJSON
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        # Convert GeoJSON to Earth Engine geometry
        geometry = ee.Geometry.MultiPolygon(geojson['features'][0]['geometry']['coordinates'])
        
        # Set default date range if not provided (January 2025)
        if start_date is None:
            start_date = '2025-01-01'
        if end_date is None:
            end_date = '2025-01-31'
        
        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.join(os.path.dirname(geojson_path), 'sentinel2_coverage.txt')
        
        print(f"\nAnalyzing Sentinel-2 coverage for date range: {start_date} to {end_date}\n")
        
        # Get Sentinel-2 coverage
        print("Fetching Sentinel-2 coverage...")
        try:
            s2_results = get_sentinel2_info(geometry, start_date, end_date)
            s2_stats = analyze_sentinel2_coverage(s2_results)
            
            # Print Sentinel-2 coverage
            print("\nSentinel-2 Coverage:")
            print("-" * 50)
            if not s2_stats:
                print("No Sentinel-2 data found")
            else:
                for mgrs_tile, data in s2_stats.items():
                    print(f"\nMGRS Tile: {mgrs_tile}")
                    if data['platforms']:
                        print(f"Platforms: {', '.join(data['platforms'])}")
                    if 'orbit_direction' in data:
                        print(f"Orbit Direction: {data['orbit_direction']}")
                    print(f"Number of acquisitions: {len(data['dates'])}")
                    if 'cloud_cover' in data:
                        print(f"Average cloud cover: {data['cloud_cover']:.2f}%")
                    print(f"Available dates: {', '.join(sorted(data['dates']))}")
        except Exception as e:
            print(f"Error fetching Sentinel-2 data: {str(e)}")
            s2_stats = {}
        
        # Save coverage information
        try:
            save_coverage_info(s2_stats, start_date, end_date, output_file)
        except Exception as e:
            print(f"Error saving coverage information: {str(e)}")
        
    except Exception as e:
        print(f"Error analyzing coverage: {str(e)}")

def main():
    """Main function"""
    try:
        # Load GeoJSON
        geojson_path = r'D:\Semester4\ProjectVijayapur\psetae\GEE-to-NPY-master\windsurf_code\geojsonfiles\croptype_KA25_wgs84_inferencing.geojson'
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        # Convert GeoJSON to Earth Engine geometry
        geometry = ee.Geometry.MultiPolygon(geojson['features'][0]['geometry']['coordinates'])
        
        # Set date range (August 2024 to March 2025)
        start_date = '2024-08-01'
        end_date = '2025-03-31'
        
        print(f"Analyzing Sentinel-2 coverage from {start_date} to {end_date}")
        print("Note: Sentinel-2 provides a 5-day revisit cycle")
        print("Note: Sentinel-2 data is gridded to MGRS tiles")
        print()
        
        # Get coverage information
        s2_results = get_sentinel2_info(geometry, start_date, end_date)
        
        # Write to file
        output_file = os.path.join(os.path.dirname(geojson_path), 'sentinel2_coverage.txt')
        write_coverage_to_file(s2_results['features'], output_file)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
