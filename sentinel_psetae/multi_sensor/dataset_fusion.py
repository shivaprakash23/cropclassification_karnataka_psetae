#
#TESTING FOR DATALOADER 
#DATASET BLOCK -  TO RETURN DATES

import torch
from torch import Tensor
from torch.utils import data

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime

import os
import json  
import random

class PixelSetData(data.Dataset):
    def __init__(self, folder, labels, npixel, sub_classes=None, norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), minimum_sampling=22, interpolate_method ='nn', return_id=False, fusion_type=None):
        """
        Args:
            folder (str): path to the main folder of the dataset, formatted as indicated in the readme
            labels (str): name of the nomenclature to use in the labels.json file
            npixel (int): Number of sampled pixels in each parcel
            sub_classes (list): If provided, only the samples from the given list of classes are considered.
            (Can be used to remove classes with too few samples)
            norm (tuple): (mean,std) tuple to use for normalization
            minimum_sampling (int): minimum number of observation to sample for Sentinel-2
            fusion_type (str): name of fusion technique to harmonize Sentinel-1 and Sentinel-2 data/features
            interpolate_method: for input-level fusion, name of method to interpolate Sentinel-1 at Sentinel-2 date
            extra_feature (str): name of the additional static feature file to use
            jitter (tuple): if provided (sigma, clip) values for the addition random gaussian noise
            return_id (bool): if True, the id of the yielded item is also returned (useful for inference)
        """
        super(PixelSetData, self).__init__()

        self.folder = folder
        self.data_folder = os.path.join(folder, 'DATA')
        self.meta_folder = os.path.join(folder, 'META')
        self.labels = labels
        self.npixel = npixel
        self.norm = norm
        self.extra_feature = extra_feature
        self.jitter = jitter  # (sigma , clip )
        self.return_id = return_id
        
        self.minimum_sampling = minimum_sampling        
        self.fusion_type = fusion_type
        self.interpolate_method = interpolate_method


        # get parcel ids
        l = [f for f in os.listdir(self.data_folder) if f.endswith('.npy')]
        self.pid = [int(f.split('.')[0]) for f in l]
        
        # Load ignored parcels if they exist
        # For inference, s2_folder should be at the same level as s1_data
        base_folder = os.path.dirname(folder)
        s2_folder = os.path.join(base_folder, 's2_data').replace('\\', '/')
        ignored_parcels_path = os.path.join(s2_folder, 'META', 'ignored_parcels.json')
        ignored_parcels = []
        if os.path.exists(ignored_parcels_path):
            with open(ignored_parcels_path, 'r') as f:
                # Read file line by line instead of using json.load
                # Each line contains a single parcel ID
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            ignored_parcels.append(int(line))
                        except ValueError:
                            print(f"Warning: Could not convert '{line}' to integer in ignored_parcels.json")
        
        # Filter out ignored parcels
        self.pid = [pid for pid in self.pid if pid not in ignored_parcels]
        self.pid = list(np.sort(self.pid))
        self.pid = list(map(str, self.pid))
        self.len = len(self.pid)

        # get Labels
        if sub_classes is not None:
            sub_indices = []
            num_classes = len(sub_classes)
            convert = dict((c, i) for i, c in enumerate(sub_classes))

        with open(os.path.join(folder, 'META', 'labels.json'), 'r') as file:
            d = json.loads(file.read())
            self.target = []
            for i, p in enumerate(self.pid):
                t = d[labels][p]

                # merge permanent(18) and temporal meadow(19)
                # this will reduce number of target classes by 1
                if t == 19:
                    t = 18

                self.target.append(t)
                if sub_classes is not None:
                    if t in sub_classes:
                        sub_indices.append(i)
                        self.target[-1] = convert[self.target[-1]]
                        
        if sub_classes is not None:
            self.pid = list(np.array(self.pid)[sub_indices])
            self.target = list(np.array(self.target)[sub_indices])
            self.len = len(sub_indices)

        # get dates for s1 and s2
        with open(os.path.join(folder, 'META', 'dates.json'), 'r') as file:
            date_s1 = json.loads(file.read())

        # For inference, s2_folder should be at the same level as s1_data
        base_folder = os.path.dirname(folder)
        s2_folder = os.path.join(base_folder, 's2_data').replace('\\', '/')
        with open(os.path.join(s2_folder, 'META', 'dates.json'), 'r') as file:
            date_s2 = json.loads(file.read())

        # for sentinel 1
        self.dates_s1 = [date_s1[i] for i in self.pid]
        self.date_positions_s1 = [date_positions(i) for i in self.dates_s1]

        # for sentinel 2
        self.dates_s2 = [date_s2[i] for i in self.pid]
        self.date_positions_s2 = [date_positions(i) for i in self.dates_s2]


        # Handle extra features
        if self.extra_feature is not None:
            with open(os.path.join(self.meta_folder, '{}.json'.format(extra_feature)), 'r') as file:
                self.extra = json.loads(file.read())

            # Convert all values to numeric lists
            processed_extra = {}
            for k, v in self.extra.items():
                if isinstance(v, (int, float)):
                    processed_extra[k] = [float(v)]
                elif isinstance(v, list):
                    # Only keep numeric values
                    numeric_list = [float(x) for x in v if isinstance(x, (int, float))]
                    if numeric_list:
                        processed_extra[k] = numeric_list
                    else:
                        processed_extra[k] = [0.0]  # Default if no numeric values
                else:
                    processed_extra[k] = [0.0]  # Default for non-numeric
            
            self.extra = processed_extra
            # Convert to numpy arrays for statistics
            values = np.array(list(processed_extra.values()))
            self.extra_m = np.mean(values, axis=0)
            self.extra_s = np.std(values, axis=0)


    # get similar day-of-year in s1 for s2
    def similar_sequence(self, input_s1, input_s2):
        input_s1 = np.asarray(input_s1)
        input_s2 = np.asarray(input_s2)

        if len(input_s1) == 0 or len(input_s2) == 0:
            # If either sequence is empty, return the first date repeated
            if len(input_s2) > 0:
                return [input_s2[0]] * len(input_s2)
            return [0] * len(input_s2)  # Default to 0 if both empty

        output_doy = []
        available_s1 = input_s1.copy()
        
        for i in input_s2:
            if len(available_s1) > 0:
                # Find closest date
                idx = np.abs(available_s1 - i).argmin()
                doy = available_s1[idx]
                output_doy.append(doy)
                # Remove used date
                available_s1 = np.delete(available_s1, idx)
            else:
                # If we run out of S1 dates, use the last matched date
                doy = output_doy[-1] if output_doy else input_s1[0]
                output_doy.append(doy)
        
        return output_doy
    

    # interpolate s1 at s2 date
    def interpolate_s1(self, arr_3d, s1_date, s2_date):
        num_pixels = arr_3d.shape[-1]
        vv = arr_3d[:,0,:]
        vh = arr_3d[:,1,:]

        # interpolate per pixel in parcel per time
        vv_interp = np.column_stack([np.interp(s2_date, s1_date, vv[:,i]) for i in range(num_pixels)])
        vh_interp = np.column_stack([np.interp(s2_date, s1_date, vh[:,i]) for i in range(num_pixels)])

        # stack vv and vh
        res = np.concatenate((np.expand_dims(vv_interp, 1), np.expand_dims(vh_interp, 1)), axis = 1)

        return res   
        
    
    def __len__(self):
        return self.len

    def __getitem__(self, item): 
        """
        Returns a Pixel-Set sequence tensor with its pixel mask and optional additional features.
        For each item npixel pixels are randomly dranw from the available pixels.
        If the total number of pixel is too small one arbitrary pixel is repeated. The pixel mask keeps track of true
        and repeated pixels.
        Returns:
              (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features) with:
                Pixel-Set: Sequence_length x Channels x npixel
                Pixel-Mask : Sequence_length x npixel
                Extra-features : Sequence_length x Number of additional features

        """
        # loader for x0 = sentinel1 and x00 = sentinel2

        x0 = np.load(os.path.join(self.folder, 'DATA', '{}.npy'.format(self.pid[item])))
        s2_folder = os.path.join(os.path.dirname(self.folder), 's2_data')
        x00 = np.load(os.path.join(s2_folder, 'DATA', '{}.npy'.format(self.pid[item])))
        y = self.target[item]
        
        s1_item_date = self.date_positions_s1[item] 
        s2_item_date = self.date_positions_s2[item] 
           
             
        # Convert date positions to numpy arrays for manipulation
        s1_item_date = np.array(s1_item_date)
        s2_item_date = np.array(s2_item_date)
        
        # Ensure consistent sequence length for S1 (16 dates)
        s1_target_len = 16
        if len(x0) > s1_target_len:
            indices = list(range(s1_target_len))
            random.shuffle(indices)
            indices = sorted(indices)
            x0 = x0[indices, :,:]
            s1_item_date = s1_item_date[indices]
        elif len(x0) < s1_target_len:
            # Pad with last observation if fewer dates
            pad_size = s1_target_len - len(x0)
            x0 = np.pad(x0, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
            s1_item_date = np.pad(s1_item_date, (0, pad_size), mode='edge')

        # Ensure consistent sequence length for S2
        s2_target_len = self.minimum_sampling if self.minimum_sampling is not None else 16
        if len(x00) > s2_target_len:
            indices = list(range(s2_target_len))
            random.shuffle(indices)
            indices = sorted(indices)
            x00 = x00[indices, :,:]
            s2_item_date = s2_item_date[indices]
        elif len(x00) < s2_target_len:
            # Pad with last observation if fewer dates
            pad_size = s2_target_len - len(x00)
            x00 = np.pad(x00, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
            s2_item_date = np.pad(s2_item_date, (0, pad_size), mode='edge')
            
        # Convert back to tensors
        s1_item_date = torch.tensor(s1_item_date)
        s2_item_date = torch.tensor(s2_item_date)
            
        
        # Handle pixel sampling for both S1 and S2
        num_pixels_s1 = x0.shape[-1]
        num_pixels_s2 = x00.shape[-1]
        
        if num_pixels_s1 == 0 or num_pixels_s2 == 0:
            # Handle empty parcels
            x = np.zeros((x0.shape[0], x0.shape[1], self.npixel))
            x2 = np.zeros((x00.shape[0], x00.shape[1], self.npixel))
            mask1, mask2 = np.zeros(self.npixel), np.zeros(self.npixel)
            mask1[0], mask2[0] = 1, 1
        else:
            # Sample or pad pixels for both sensors
            if num_pixels_s1 >= self.npixel:
                idx_s1 = np.random.choice(num_pixels_s1, size=self.npixel, replace=False)
                x = x0[:, :, idx_s1]
                mask1 = np.ones(self.npixel)
            else:
                x = np.zeros((x0.shape[0], x0.shape[1], self.npixel))
                x[:, :, :num_pixels_s1] = x0
                x[:, :, num_pixels_s1:] = np.stack([x0[:, :, 0] for _ in range(num_pixels_s1, self.npixel)], axis=-1)
                mask1 = np.array([1] * num_pixels_s1 + [0] * (self.npixel - num_pixels_s1))
            
            if num_pixels_s2 >= self.npixel:
                idx_s2 = np.random.choice(num_pixels_s2, size=self.npixel, replace=False)
                x2 = x00[:, :, idx_s2]
                mask2 = np.ones(self.npixel)
            else:
                x2 = np.zeros((x00.shape[0], x00.shape[1], self.npixel))
                x2[:, :, :num_pixels_s2] = x00
                x2[:, :, num_pixels_s2:] = np.stack([x00[:, :, 0] for _ in range(num_pixels_s2, self.npixel)], axis=-1)
                mask2 = np.array([1] * num_pixels_s2 + [0] * (self.npixel - num_pixels_s2))

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m)
            s = np.array(s)

            if len(m.shape) == 0:
                x = (x - m) / s
            elif len(m.shape) == 1:  # Normalise channel-wise
                x = (x.swapaxes(1, 2) - m) / s
                x = x.swapaxes(1, 2)  # Normalise channel-wise for each date
            elif len(m.shape) == 2:
                x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
                x = (x - m) / s
                x = np.swapaxes((np.rollaxis(x, 1)), 1, 2)
                
        x = x.astype('float')
        x2 = x2.astype('float')

        if self.jitter is not None:
            sigma, clip = self.jitter
            x = x + np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)
            x2 = x2 + np.clip(sigma * np.random.randn(*x2.shape), -1 * clip, clip)

        mask1 = np.stack([mask1 for _ in range(x.shape[0])], axis=0)  # Add temporal dimension to mask
        mask2 = np.stack([mask2 for _ in range(x2.shape[0])], axis=0)


        # interpolate s1 at s2 date
        if self.fusion_type == 'early' or self.fusion_type == 'pse':
        
            if self.interpolate_method == 'nn':
                output_doy = self.similar_sequence(input_s1 = s1_item_date, input_s2 = s2_item_date)

                # Convert to numpy arrays for comparison
                s1_positions = np.array(self.date_positions_s1[item])
                output_doy = np.array(output_doy)
                
                # Find indices where s1 dates match output_doy
                x_idx = np.where(np.isin(s1_positions, output_doy))[0]
                
                # Select matching timesteps
                x = x[x_idx, :, :]
                mask1 = mask1[x_idx,:]
            
            elif self.interpolate_method == 'linear':
                x = self.interpolate_s1(arr_3d = x, s1_date = s1_item_date, s2_date = s2_item_date)
                mask1 = mask1[:len(s2_item_date), :] # slice to length of s2_sequence

        
        # create tensor from numpy
        data = (Tensor(x), Tensor(mask1))
        data2 = (Tensor(x2), Tensor(mask2))

        # Convert dates to tensors and normalize
        s1_dates = Tensor(np.array(self.date_positions_s1[item])).long()
        s2_dates = Tensor(np.array(self.date_positions_s2[item])).long()

        if self.extra_feature is not None:
            ef = (self.extra[str(self.pid[item])] - self.extra_m) / self.extra_s

        # interpolate s1 at s2 date
        if self.fusion_type == 'early' or self.fusion_type == 'pse':
        
            if self.interpolate_method == 'nn':
                output_doy = self.similar_sequence(input_s1 = s1_item_date, input_s2 = s2_item_date)

                # Convert to numpy arrays for comparison
                s1_positions = np.array(self.date_positions_s1[item])
                output_doy = np.array(output_doy)
                
                # Find indices where s1 dates match output_doy
                x_idx = np.where(np.isin(s1_positions, output_doy))[0]
                
                # Select matching timesteps
                x = x[x_idx, :, :]
                mask1 = mask1[x_idx,:]
            
            elif self.interpolate_method == 'linear':
                x = self.interpolate_s1(arr_3d = x, s1_date = s1_item_date, s2_date = s2_item_date)
                mask1 = mask1[:len(s2_item_date), :] # slice to length of s2_sequence

        # create tensor from numpy
        data = (Tensor(x), Tensor(mask1))
        data2 = (Tensor(x2), Tensor(mask2))

        # Convert dates to tensors and normalize
        s1_dates = Tensor(np.array(s1_item_date)).long()
        s2_dates = Tensor(np.array(s2_item_date)).long()

        if self.extra_feature is not None:
            ef = (self.extra[str(self.pid[item])] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()
            ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)

        # Convert target to int64 tensor
        target = torch.tensor(self.target[item], dtype=torch.int64)

        if self.return_id:
            return data, data2, target, (s1_dates, s2_dates), self.pid[item]
        else:
            return data, data2, target, (s1_dates, s2_dates)


class PixelSetData_preloaded(PixelSetData):
    """ Wrapper class to load all the dataset to RAM at initialization (when the hardware permits it).
    """
    def __init__(self, folder, labels, npixel, sub_classes=None, norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), minimum_sampling=22, interpolate_method ='nn', return_id=False, fusion_type=None):
        super(PixelSetData_preloaded, self).__init__(folder, labels, npixel, sub_classes, norm, extra_feature, jitter,                           minimum_sampling, interpolate_method, return_id, fusion_type)
        
        self.samples = []
        print('Loading samples to memory . . .')
        for item in range(len(self)):
            self.samples.append(super(PixelSetData_preloaded, self).__getitem__(item))
        print('Done !')

    def __getitem__(self, item):
        return self.samples[item]


def parse(date):
    d = str(date)
    return int(d[:4]), int(d[4:6]), int(d[6:])


def interval_days(date1, date2):
    return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)


def date_positions(dates):
    pos = []
    if not dates:
        return pos
    
    # Calculate days from first date and normalize to be within T=366 (full year)
    for d in dates:
        days = interval_days(d, dates[0])
        normalized_days = days % 366  # Using 366 for a full year (including leap year)
        pos.append(normalized_days)
    
    # Convert to tensor
    if pos:
        pos = torch.tensor(pos)
    return pos
    pos = []
    for d in dates:
        days = interval_days(d, dates[0])
        # Normalize to be within 0-359 range (T=360 in config)
        normalized_days = days % 360
        pos.append(normalized_days)
    return pos
