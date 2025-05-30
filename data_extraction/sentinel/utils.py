import ee
import os
import argparse
import json
from tqdm import tqdm
from datetime import datetime

ee.Authenticate()
ee.Initialize(project='ee-shivaprakashssy-psetae-ka28')


def get_collection(geometry, col_id, start_date , end_date, num_per_month, cloud_cover, addNDVI, footprint_id, speckle_filter, kernel_size):
    print("col_id that is passed",col_id)
    if 'S2' in col_id: 
        collection = ee.ImageCollection(col_id).filterDate(start_date,end_date).filterBounds(geometry).filter(
                     ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',cloud_cover)).select(
                     ['B2','B3','B4','B5', 'B6','B7','B8','B8A','B11','B12'])
        #print("col_id that is passed",col_id)
        #print("After collection:", collection.size().getInfo())
        #print("footprint_id that is passed",footprint_id)  
        if footprint_id is not None:
            collection = collection.filter(ee.Filter.inList('MGRS_TILE', ee.List(footprint_id)))
        
        #print('After footprint_id', collection.size().getInfo())
        # compute NDVI
        if addNDVI:
            collection = collection.map(lambda img: ee.Image(img).addBands(img.normalizedDifference(['B8', 'B4']).rename('ndvi')))

        # get normalisation statistics (placed prior to any parcel clipping operation)
        collection = collection.map(lambda img: img.set('stats', ee.Image(img).reduceRegion(reducer=ee.Reducer.percentile([2, 98]), bestEffort=True)))

        #print("After collection:", collection.size().getInfo())                 
            
    elif 'S1'  in col_id:
        collection = ee.ImageCollection(col_id).filter(ee.Filter.eq('instrumentMode', 'IW')).filterDate(
                     start_date, end_date).filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).filter(
                     ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).filterBounds(geometry).select(['VV','VH']).filter(
                     ee.Filter.eq('orbitProperties_pass', 'DESCENDING')).sort('system:time_start', True)
        
        if footprint_id is not None:
            collection = collection.filter(ee.Filter.inList('relativeOrbitNumber_start', ee.List(footprint_id)))            
            
        # get normalisation statistics (placed prior to any parcel clipping operation)
        collection = collection.map(lambda img: img.set('stats', ee.Image(img).reduceRegion(reducer=ee.Reducer.percentile([2, 98]), bestEffort=True)))
         
        # clip using 1km buffer of geometry to avoid excessive computation in speckle filtering
        collection = collection.map(lambda img: ee.Image(img).clip(geometry.bounds().buffer(1000)))
        
        # multi-temporal speckle reduction
        if speckle_filter == 'temporal':
            collection = multitemporalDespeckle(collection, kernel_size, units ='pixels', opt_timeWindow={'before': -2, 'after': 2, 'units': 'month'})

        # focal mean
        elif speckle_filter == 'mean':
            collection = collection.map(lambda img: ee.Image(img).focal_mean(radius = kernel_size, kernelType = 'square', units='pixels').copyProperties(img, ["system:time_start", "stats"]))

         # focal median                                            
        elif speckle_filter == 'median':
            collection = collection.map(lambda img: ee.Image(img).focal_median(radius = kernel_size, kernelType = 'square', units='pixels').copyProperties(img, ["system:time_start", "stats"]))                                           

        #  co-register Sentinel-1 & Sentinel-2
        # Using UTM Zone 43N (EPSG:32643) which is appropriate for Karnataka, India (around 76°E, 16°N)
        collection = collection.map(lambda img: ee.Image(img).reproject(crs = 'EPSG:32643', crsTransform = [10, 0, 500000, 0, -10, 3000000]))

    print("After s1 and s2 :")

    # checks for partly-covered and duplicate footprints and clip collection to geometry
    collection = overlap_filter(collection, geometry)
    
    print("After overlap:")
    
    # return one image per month
    if  num_per_month > 0:
        collection = monthly_(col_id, collection, start_year = int(start_date[:4]), end_year = int(end_date[:4]), num_per_month=num_per_month)
    
    print("After num_per_month:")
    return collection



def monthly_(col_id, collection, start_year, end_year, num_per_month):
    """
    description:
        return n images per month for a given year sequence
    """    
    months = ee.List.sequence(1, 12)
    years = ee.List.sequence(start_year, end_year)

    try:
        if 'S2' in col_id: 
            collection = ee.ImageCollection.fromImages(years.map(lambda y: months.map(lambda m: collection.filter(
                        ee.Filter.calendarRange(y, y, 'year')).filter(ee.Filter.calendarRange(m, m, 'month')).sort(
                        'CLOUDY_PIXEL_PERCENTAGE').toList(num_per_month))).flatten())
            
            # sort by doa for ordered date sequence
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
    # creates output directory
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'DATA'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'META'), exist_ok=True)


def parse_rpg(rpg_file, label_names=['CODE_GROUP'], id_field = 'ID_PARCEL'):
    """Reads rpg and returns a dict of pairs (ID_PARCEL : Polygon) and a dict of dict of labels
     {label_name1: {(ID_PARCEL : Label value)},
      label_name2: {(ID_PARCEL : Label value)}
     }
     """
    # Read rpg file
    print('Reading RPG . . .')
    with open(rpg_file) as f:
        data = json.load(f)
    print('reading polygons')
    # Get list of polygons
    polygons = {}
    lab_rpg = dict([(l, {}) for l in label_names])

    for f in tqdm(data['features']):
        # p = Polygon(f['geometry']['coordinates'][0][0])
        p = f["geometry"]["coordinates"][0]  
        polygons[f['properties'][id_field]] = p
        for l in label_names:
            lab_rpg[l][f['properties'][id_field]] = f['properties'][l]
            #print(polygons)
    return polygons, lab_rpg


# def shapely2ee(geometry):
#     # converts geometry to GEE server object
#     pt_list = list(zip(*geometry.exterior.coords.xy))
#     return ee.Geometry.Polygon(pt_list)


def geom_features(geometry):
    # computes geometric info per parcel
    area  = geometry.area().getInfo()
    perimeter = geometry.perimeter().getInfo()
    bbox = geometry.bounds()
    return perimeter, perimeter/area, bbox


def overlap_filter(collection, geometry):

    # set masked/no data pixels to -9999
    collection = collection.filterBounds(geometry).map(lambda image: ee.Image(image).unmask(-9999).clip(geometry))
    
    #add image properties {doa, noData & overlap assertions}
    collection = collection.map(lambda image: image.set({
        'doa': ee.Date(image.get('system:time_start')).format('YYYYMMdd'),
        'noData': ee.Image(image).clip(geometry).reduceRegion(ee.Reducer.toList(), geometry).values().flatten().contains(-9999),
        'overlap': ee.Image(image).geometry().contains(geometry, 0.01)}))
    
    # remove tiles containing masked pixels, select one of many overlapping tiles over a parcel
    collection = collection.filter(ee.Filter.eq('noData', False)).filter(ee.Filter.eq('overlap',True)).distinct('doa')
                                            
    return collection

# min-max normalisation using 2 & 98 percentile
def normalize(img):
    img = ee.Image(img)
    def norm_band(name):
        name = ee.String(name)
        stats = ee.Dictionary(img.get('stats'))
        p2 = ee.Number(stats.get(name.cat('_p2')))
        p98 = ee.Number(stats.get(name.cat('_p98')))
        stats_img = img.select(name).subtract(p2).divide((p98.subtract(p2)))
        return stats_img
    
    new_img = img.addBands(srcImg = ee.ImageCollection.fromImages(img.bandNames().map(norm_band)).toBands().rename(img.bandNames()), overwrite=True)
    return new_img.toFloat()



def multitemporalDespeckle(images, kernel_size, units ='pixels', opt_timeWindow={'before': -2, 'after': 2, 'units': 'month'}):

    bandNames = ee.Image(images.first()).bandNames()
    bandNamesMean = bandNames.map(lambda b: ee.String(b).cat('_mean'))
    bandNamesRatio = bandNames.map(lambda b: ee.String(b).cat('_ratio'))

    # compute space-average for all images
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

    # denoise images
    return meanSpace.map(multitemporalDespeckleSingle).select(bandNames)



def parse_args():
    parser = argparse.ArgumentParser(description='Query GEE for time series data and return numpy array per parcel')
                                            
    # parcels geometryies (json)
    parser.add_argument('rpg_file', type=str, help="path to json with attributes ID_PARCEL, CODE_GROUP")                                        
    parser.add_argument('--id_field', type=str, default='ID_PARCEL', nargs="?", help='parcel id column name in json file')
    parser.add_argument('--label_names', type=list, default=['CODE_GROUP'], nargs="?", help='label column name in json file')    
                                            
    # GEE params
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('--col_id', type=str, default="COPERNICUS/S2_SR", nargs="?", help="GEE collection ID e.g. 'COPERNICUS/S2_SR' or 'COPERNICUS/S1_GRD'")
    parser.add_argument('--start_date', type=str,  default='2018-10-01', nargs="?", help='start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str,  default='2019-12-31', nargs="?", help='end date YYYY-MM-DD')
    parser.add_argument('--num_per_month', type=int, default=0, nargs="?", help='number of scenes per month. if 0 returns all')
    parser.add_argument('--footprint_id', type=list, default=None, nargs="?", help='granule/orbit identifier for Sentinel-1 eg [153, 154] or Sentinel-2 eg ["30UUU"]')  
                                            
    # Sentinel-1
    parser.add_argument('--speckle_filter', type=str, default='temporal', nargs="?", help='reduce speckle using multi-temporal despeckling. options = [temporal, mean, median]')    
    parser.add_argument('--kernel_size', type=int, default =5, nargs="?", help='kernel/window size in pixels for despeckling')                                           
   
    # Sentinel-2                                          
    parser.add_argument('--cloud_cover', type=int, default=80, nargs="?", help='cloud cover threshold')  
    parser.add_argument('--addNDVI', type=bool, default=False, nargs="?", help='computes and append ndvi as additional band')  
    
    return parser.parse_args()
