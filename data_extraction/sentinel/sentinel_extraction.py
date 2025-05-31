import ee
import os
import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
import warnings
import argparse

def get_collection(geometry, col_id, start_date, end_date, num_per_month, cloud_cover, addNDVI, footprint_id, speckle_filter, kernel_size):
    """Get Sentinel collection filtered by date, bounds, and other parameters"""
    print("col_id that is passed", col_id)
    if 'S2' in col_id: 
        collection = ee.ImageCollection(col_id).filterDate(start_date, end_date).filterBounds(geometry).filter(
                     ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)).select(
                     ['B2','B3','B4','B5', 'B6','B7','B8','B8A','B11','B12'])
        
        if footprint_id is not None:
            collection = collection.filter(ee.Filter.inList('MGRS_TILE', ee.List(footprint_id)))
        
        if addNDVI:
            collection = collection.map(lambda img: ee.Image(img).addBands(img.normalizedDifference(['B8', 'B4']).rename('ndvi')))

        collection = collection.map(lambda img: img.set('stats', ee.Image(img).reduceRegion(reducer=ee.Reducer.percentile([2, 98]), bestEffort=True)))
                     
    elif 'S1' in col_id:
        collection = ee.ImageCollection(col_id).filter(ee.Filter.eq('instrumentMode', 'IW')).filterDate(
                     start_date, end_date).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(
                     ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filterBounds(geometry).select(['VV','VH']).filter(
                     ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).sort('system:time_start', True)
        
        if footprint_id is not None:
            collection = collection.filter(ee.Filter.inList('relativeOrbitNumber_start', ee.List(footprint_id)))            
            
        collection = collection.map(lambda img: img.set('stats', ee.Image(img).reduceRegion(reducer=ee.Reducer.percentile([2, 98]), bestEffort=True)))
        collection = collection.map(lambda img: ee.Image(img).clip(geometry.bounds().buffer(1000)))
        
        if speckle_filter == 'temporal':
            collection = multitemporalDespeckle(collection, kernel_size, units='pixels', opt_timeWindow={'before': -2, 'after': 2, 'units': 'month'})
        elif speckle_filter == 'mean':
            collection = collection.map(lambda img: ee.Image(img).focal_mean(radius=kernel_size, kernelType='square', units='pixels').copyProperties(img, ["system:time_start", "stats"]))
        elif speckle_filter == 'median':
            collection = collection.map(lambda img: ee.Image(img).focal_median(radius=kernel_size, kernelType='square', units='pixels').copyProperties(img, ["system:time_start", "stats"]))                                           

        collection = collection.map(lambda img: ee.Image(img).reproject(crs='EPSG:32643', crsTransform=[10, 0, 500000, 0, -10, 3000000]))

    print("After s1 and s2 :")
    collection = overlap_filter(collection, geometry)
    print("After overlap:")
    
    if num_per_month > 0:
        collection = monthly_(col_id, collection, start_year=int(start_date[:4]), end_year=int(end_date[:4]), num_per_month=num_per_month)
    
    print("After num_per_month:")
    return collection

def monthly_(col_id, collection, start_year, end_year, num_per_month):
    """Return n images per month for a given year sequence"""    
    months = ee.List.sequence(1, 12)
    years = ee.List.sequence(start_year, end_year)

    try:
        if 'S2' in col_id: 
            collection = ee.ImageCollection.fromImages(years.map(lambda y: months.map(lambda m: collection.filter(
                        ee.Filter.calendarRange(y, y, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).sort(
                        'CLOUDY_PIXEL_PERCENTAGE').toList(num_per_month))).flatten())
            collection = collection.sort('system:time_start')
                
        elif 'S1' in col_id: 
            collection = ee.ImageCollection.fromImages(years.map(lambda y: months.map(lambda m: collection.filter(
                        ee.Filter.calendarRange(y, y, 'year')).filter(ee.Filter.calendarRange(m, m, 'month'))
                        .toList(num_per_month))).flatten())
            collection = collection.sort('system:time_start')
            
        return collection
    except:
        print("collection cannot be filtered")

def prepare_output(output_path):
    """Creates output directory structure"""
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'META'), exist_ok=True)

def parse_rpg(rpg_file, label_names=['CODE_GROUP'], id_field='ID_PARCEL'):
    """Reads rpg and returns a dict of pairs (ID_PARCEL : Polygon) and a dict of dict of labels"""
    print('Reading RPG . . .')
    with open(rpg_file) as f:
        data = json.load(f)
    print('reading polygons')
    polygons = {}
    lab_rpg = dict([(l, {}) for l in label_names])

    for f in tqdm(data['features']):
        p = f["geometry"]["coordinates"][0]  
        polygons[f['properties'][id_field]] = p
        for l in label_names:
            lab_rpg[l][f['properties'][id_field]] = f['properties'][l]
    return polygons, lab_rpg

def geom_features(geometry):
    """Computes geometric info per parcel"""
    area = geometry.area().getInfo()
    perimeter = geometry.perimeter().getInfo()
    bbox = geometry.bounds()
    return perimeter, perimeter/area, bbox

def overlap_filter(collection, geometry):
    """Filter collection based on overlap with geometry"""
    collection = collection.filterBounds(geometry).map(lambda image: ee.Image(image).unmask(-9999).clip(geometry))
    
    collection = collection.map(lambda image: image.set({
        'doa': ee.Date(image.get('system:time_start')).format('YYYYMMdd'),
        'noData': ee.Image(image).clip(geometry).reduceRegion(ee.Reducer.toList(), geometry).values().flatten().contains(-9999),
        'overlap': ee.Image(image).geometry().contains(geometry, 0.01)}))
    
    collection = collection.filter(ee.Filter.eq('noData', False)).filter(ee.Filter.eq('overlap',True)).distinct('doa')
    return collection

def normalize(img):
    """Min-max normalisation using 2 & 98 percentile"""
    img = ee.Image(img)
    def norm_band(name):
        name = ee.String(name)
        stats = ee.Dictionary(img.get('stats'))
        p2 = ee.Number(stats.get(name.cat('_p2')))
        p98 = ee.Number(stats.get(name.cat('_p98')))
        stats_img = img.select(name).subtract(p2).divide((p98.subtract(p2)))
        return stats_img
    
    new_img = img.addBands(srcImg=ee.ImageCollection.fromImages(img.bandNames().map(norm_band)).toBands().rename(img.bandNames()), overwrite=True)
    return new_img.toFloat()

def multitemporalDespeckle(images, kernel_size, units='pixels', opt_timeWindow={'before': -2, 'after': 2, 'units': 'month'}):
    """Multi-temporal speckle filter for Sentinel-1 data"""
    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    def space_avg(image):
        mean = image.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(kernel_size, units)).rename(bandNamesMean)
        ratio = image.divide(mean).rename(bandNamesRatio)
        return image.addBands(mean).addBands(ratio)

    meanSpace = images.map(space_avg)

    def multitemporalDespeckleSingle(image):
        t = ee.Image(image).date()
        start = t.advance(ee.Number(opt_timeWindow['before']), opt_timeWindow['units'])
        end = t.advance(ee.Number(opt_timeWindow['after']), opt_timeWindow['units'])
        meanSpace2 = ee.ImageCollection(meanSpace).select(bandNamesRatio).filterDate(start, end)
        b = image.select(bandNamesMean)
        return b.multiply(meanSpace2.sum()).divide(meanSpace2.size()).rename(bandNames).copyProperties(image, ['system:time_start', 'stats']) 

    return meanSpace.map(multitemporalDespeckleSingle).select(bandNames)

def prepare_dataset(rpg_file, label_names, id_field, output_dir, col_id, start_date, end_date, num_per_month, 
                   cloud_cover, addNDVI, footprint_id, speckle_filter, kernel_size):
    """Main function to prepare the dataset from Sentinel data"""
    warnings.filterwarnings('error', category=DeprecationWarning)
    start = datetime.now()

    prepare_output(output_dir)
    polygons, lab_rpg = parse_rpg(rpg_file, label_names, id_field)

    dates = {k:[] for k in list(polygons.keys())}
    labels = dict([(l, {}) for l in lab_rpg.keys()])
    geom_feats = {k: {} for k in list(polygons.keys())}
    print("dictionary created")
    ignored = 0

    for parcel_id, geometry in tqdm(polygons.items()):
        geometry = ee.Geometry.Polygon(geometry)
        collection = get_collection(geometry, col_id, start_date, end_date, num_per_month, 
                                 cloud_cover, addNDVI, footprint_id, speckle_filter, kernel_size)
        collection = collection.map(normalize)
        
        collection = collection.map(lambda img: img.set('temporal', 
                                 ee.Image(img).reduceRegion(reducer=ee.Reducer.toList(), 
                                                         geometry=geometry, scale=10).values()))
        print("collection done")

        try:
            np_all_dates = np.array(collection.aggregate_array('temporal').getInfo())
            print("Info on temporal")
            assert np_all_dates.shape[-1] > 0 
            
        except:
            print('Error in parcel --------------------> {}'.format(parcel_id))
            with open(os.path.join(output_dir, 'META', 'ignored_parcels.json'), 'a+') as file:
                file.write(json.dumps(int(parcel_id))+'\n')
            ignored += 1
            
        else:
            date_series = collection.aggregate_array('doa').getInfo()
            dates[str(parcel_id)] = date_series

            for l in labels.keys():
                labels[l][parcel_id] = int(lab_rpg[l][parcel_id])
            
            perimeter, shape_ratio, bbox = geom_features(geometry)
            geom_feats[str(parcel_id)] = [int(perimeter)]
            print(geom_feats[str(parcel_id)])
            
            np.save(os.path.join(output_dir, 'DATA', str(parcel_id)), np_all_dates)

        with open(os.path.join(output_dir, 'META', 'geomfeat.json'), 'w') as file:
            file.write(json.dumps(geom_feats, indent=4))

        with open(os.path.join(output_dir, 'META', 'labels.json'), 'w') as file:
            file.write(json.dumps(labels, indent=4))

        with open(os.path.join(output_dir, 'META', 'dates.json'), 'w') as file:
            file.write(json.dumps(dates, indent=4))

    end = datetime.now()
    print('total ignored parcels', ignored)
    print(f"\n processing time -> {end - start}")

def main():
    parser = argparse.ArgumentParser(description='Extract Sentinel-1 or Sentinel-2 time series data from Google Earth Engine')
    
    parser.add_argument('rpg_file', type=str, help="path to json with attributes ID_PARCEL, CODE_GROUP")                                        
    parser.add_argument('--id_field', type=str, default='ID_PARCEL', help='parcel id column name in json file')
    parser.add_argument('--label_names', type=list, default=['CODE_GROUP'], help='label column name in json file')    
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('--col_id', type=str, default="COPERNICUS/S2_SR", help="GEE collection ID e.g. 'COPERNICUS/S2_SR' or 'COPERNICUS/S1_GRD'")
    parser.add_argument('--start_date', type=str, default='2024-09-01', help='start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default='2025-03-31', help='end date YYYY-MM-DD')
    parser.add_argument('--num_per_month', type=int, default=0, help='number of scenes per month. if 0 returns all')
    parser.add_argument('--footprint_id', type=list, default=None, help='granule/orbit identifier for Sentinel-1 eg [153, 154] or Sentinel-2 eg ["30UUU"]')  
    parser.add_argument('--speckle_filter', type=str, default='temporal', help='reduce speckle using multi-temporal despeckling. options = [temporal, mean, median]')    
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel/window size in pixels for despeckling')                                           
    parser.add_argument('--cloud_cover', type=int, default=80, help='cloud cover threshold')  
    parser.add_argument('--addNDVI', type=bool, default=False, help='computes and append ndvi as additional band')  
    
    args = parser.parse_args()

    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project='your-project-id')  # Replace with your GEE project ID

    prepare_dataset(
        rpg_file=args.rpg_file,
        label_names=args.label_names,
        id_field=args.id_field,
        output_dir=args.output_dir,
        col_id=args.col_id,
        start_date=args.start_date,
        end_date=args.end_date,
        num_per_month=args.num_per_month,
        cloud_cover=args.cloud_cover,
        addNDVI=args.addNDVI,
        footprint_id=args.footprint_id,
        speckle_filter=args.speckle_filter,
        kernel_size=args.kernel_size
    )

if __name__ == "__main__":
    main()
