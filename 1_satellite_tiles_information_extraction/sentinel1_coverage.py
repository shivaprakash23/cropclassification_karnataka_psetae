import ee
import os
from datetime import datetime, timedelta
import json
from collections import defaultdict

# Initialize Earth Engine with your project
ee.Initialize(project='ee-shivaprakashssy-psetae')

def get_sentinel1_info(geometry, start_date, end_date, orbit_pass):
    """Get Sentinel-1 coverage information"""
    # Print the geometry centroid for debugging
    try:
        centroid = geometry.centroid().coordinates().getInfo()
        print(f"Geometry centroid: {centroid}")
    except Exception as e:
        print(f"Error getting centroid: {str(e)}")
    
    # Print the date range and orbit pass for debugging
    print(f"Searching for Sentinel-1 {orbit_pass} passes from {start_date} to {end_date}")
    
    try:
        # Get all platforms first without date filtering
        all_collection = ee.ImageCollection('COPERNICUS/S1_GRD')
        all_platforms = all_collection.filterBounds(geometry) \
            .aggregate_array('platform_number').distinct().getInfo()
        print(f"All available platforms in this region (all time): {all_platforms}")
        
        # Now filter by date
        date_only_collection = all_collection.filterDate(start_date, end_date)
        date_count = date_only_collection.size().getInfo()
        print(f"Sentinel-1 images available for this date range (globally): {date_count}")
        
        # Filter by orbit pass (DESCENDING only)
        orbit_collection = date_only_collection.filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
        orbit_count = orbit_collection.size().getInfo()
        print(f"Sentinel-1 {orbit_pass} passes for this date range (globally): {orbit_count}")
        
        # Create filters for S1A and S1C
        s1a_filter = ee.Filter.eq('platform_number', 'A')
        s1c_filter = ee.Filter.eq('platform_number', 'C')
        combined_filter = ee.Filter.Or([s1a_filter, s1c_filter])
        
        # Filter by geometry first
        collection = orbit_collection.filterBounds(geometry)
        
        # Print available platforms before filtering
        platforms = collection.aggregate_array('platform_number').distinct().getInfo()
        print(f"Available platforms in this region: {platforms}")
        
        # Filter by S1A/S1C platforms
        collection = collection.filter(combined_filter)
        
        # Get the count of images for this location
        location_count = collection.size().getInfo()
        print(f"Sentinel-1 {orbit_pass} passes for this location and date range: {location_count}")
        
        if location_count == 0:
            print(f"No Sentinel-1 {orbit_pass} passes found for the specified area and date range")
            return None
        
        # Get metadata for each image
        metadata = collection.map(lambda img: ee.Feature(None, {
            'date': ee.Date(img.date()).format('YYYY-MM-dd'),
            'platform': img.get('platform_number'),
            'orbit_pass': img.get('orbitProperties_pass'),
            'track': img.get('relativeOrbitNumber_start'),
            'instrument_mode': img.get('instrumentMode')
        }))
        
        # Print available instrument modes
        modes = collection.aggregate_array('instrumentMode').distinct().getInfo()
        print(f"Available instrument modes: {modes}")
        
        return metadata.getInfo()
    except Exception as e:
        print(f"Error in Sentinel-1 data retrieval: {str(e)}")
        return None

def analyze_sentinel1_coverage(results):
    """Analyze Sentinel-1 coverage statistics"""
    track_data = defaultdict(lambda: {'dates': [], 'platforms': set(), 'orbit_pass': set(), 'relative_orbits': set()})
    
    if results is None or 'features' not in results:
        print("Warning: No Sentinel-1 data found for the given parameters")
        return track_data

    print(f"Processing {len(results['features'])} Sentinel-1 features")
    
    for feature in results['features']:
        props = feature['properties']
        date = props.get('date')
        platform = props.get('platform')
        orbit_pass = props.get('orbit_pass')
        relative_orbit = props.get('relative_orbit')
        track = props.get('track')

        # Debug individual feature
        print(f"Feature: Date={date}, Platform={platform}, Track={track}, Orbit={relative_orbit}, Pass={orbit_pass}")
        
        # Use 'unknown' for missing track numbers instead of None
        track_key = str(track) if track is not None else 'unknown'
        
        if date:
            track_data[track_key]['dates'].append(date)
            if platform:
                track_data[track_key]['platforms'].add(f"S1{platform}")
            if orbit_pass:
                track_data[track_key]['orbit_pass'].add(orbit_pass)
            if relative_orbit:
                track_data[track_key]['relative_orbits'].add(relative_orbit)

    # Print summary of what we found
    print(f"Found data for {len(track_data)} tracks")
def write_coverage_to_file(features, output_file):
    """Write coverage information to a file"""
    # Group features by track
    tracks = defaultdict(list)
    for feature in features:
        track = feature['properties'].get('track', 'unknown')
        tracks[track].append(feature)
    
    with open(output_file, 'w') as f:
        f.write("Sentinel-1 Coverage Analysis\n")
        f.write("======================\n\n")
        
        f.write(f"Found data for {len(tracks)} tracks\n")
        for track, track_features in tracks.items():
            f.write(f"Track {track}: {len(track_features)} acquisitions\n")
        f.write("\n")
        
        for track, features in tracks.items():
            # Calculate monthly statistics
            monthly_stats = defaultdict(lambda: {'count': 0})
            yearly_stats = defaultdict(lambda: {'count': 0})
            
            for feature in features:
                date_obj = datetime.strptime(feature['properties']['date'], '%Y-%m-%d')
                month_key = (date_obj.year, date_obj.month)
                year_key = date_obj.year
                
                # Update monthly stats
                monthly_stats[month_key]['count'] += 1
                
                # Update yearly stats
                yearly_stats[year_key]['count'] += 1
            
            f.write(f"Detailed Coverage for Track {track}\n")
            f.write("-" * 40 + "\n")
            f.write("Year | Month | Date       | Platform  | Pass Direction\n")
            f.write("-" * 55 + "\n")
            for feature in sorted(features, key=lambda x: x['properties']['date']):
                date_obj = datetime.strptime(feature['properties']['date'], '%Y-%m-%d')
                year = date_obj.year
                month = date_obj.strftime('%b')
                platform = f"S1{feature['properties']['platform']}"
                orbit_pass = feature['properties']['orbit_pass']
                f.write(f"{year} | {month:>5} | {feature['properties']['date']} | {platform:<9} | {orbit_pass}\n")
            f.write("\n")
            
            # Write monthly statistics
            f.write("Monthly Statistics\n")
            f.write("-" * 40 + "\n")
            f.write("Year | Month | Acquisitions\n")
            f.write("-" * 45 + "\n")
            for (year, month), stats in sorted(monthly_stats.items()):
                month_name = datetime(year, month, 1).strftime('%b')
                f.write(f"{year} | {month_name:>5} | {stats['count']:>12}\n")
            f.write("\n")
            
            # Write yearly statistics
            f.write("Yearly Statistics\n")
            f.write("-" * 40 + "\n")
            f.write("Year | Acquisitions\n")
            f.write("-" * 40 + "\n")
            for year, stats in sorted(yearly_stats.items()):
                f.write(f"{year} | {stats['count']:>12}\n")
            f.write("\n")
        
        # Write footer with time range
        all_dates = [datetime.strptime(f['properties']['date'], '%Y-%m-%d') for f in features]
        if all_dates:
            start = min(all_dates).strftime('%Y-%m-%d')
            end = max(all_dates).strftime('%Y-%m-%d')
            f.write(f"Time Range: {start} to {end}\n")
            f.write(f"Total Acquisitions: {len(features)}\n")
    
    print(f"\nCoverage information saved to: {os.path.abspath(output_file)}")

def main():
    """Main function"""
    try:
        # Load GeoJSON
        geojson_path = r'D:\Semester4\ProjectVijayapur\psetae\GEE-to-NPY-master\windsurf_code\geojsonfiles\croptype_KA28_fortileextraction.geojson'
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        # Convert GeoJSON to Earth Engine geometry
        geometry = ee.Geometry.MultiPolygon(geojson['features'][0]['geometry']['coordinates'])
        
        # Set date range (August 2024 to March 2025)
        start_date = '2024-08-01'
        end_date = '2025-03-31'
        
        print(f"\nAnalyzing Sentinel-1 coverage from {start_date} to {end_date}")
        print("Note: Sentinel-1 provides a 12-day revisit cycle")
        print("Note: Sentinel-1 data is organized by tracks\n")
        
        # Get coverage for both ascending and descending passes
        ascending_results = get_sentinel1_info(geometry, start_date, end_date, 'ASCENDING')
        descending_results = get_sentinel1_info(geometry, start_date, end_date, 'DESCENDING')
        
        # Combine results
        combined_features = []
        if ascending_results and 'features' in ascending_results:
            combined_features.extend(ascending_results['features'])
            print(f"Found {len(ascending_results['features'])} ascending passes")
        if descending_results and 'features' in descending_results:
            combined_features.extend(descending_results['features'])
            print(f"Found {len(descending_results['features'])} descending passes")
        
        # Write coverage information to file
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentinel1_coverage.txt')
        if combined_features:
            write_coverage_to_file(combined_features, output_file)
            print(f"\nTotal passes found: {len(combined_features)}")
        else:
            print('No Sentinel-1 data found for the specified parameters')

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
