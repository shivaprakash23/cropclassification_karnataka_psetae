import torch
from torch import Tensor
from torch.utils import data
import torch.nn.functional as F

import pandas as pd
import numpy as np
import datetime as dt

import os
import json
import random

def custom_collate_fn(batch):
    """Custom collate function to handle variable sequence lengths.
    Args:
        batch: List of tuples ((x, mask), y, dates)
    Returns:
        Padded batch with consistent sequence length
    """
    # Split batch into components
    batch_x_mask = []
    batch_y = []
    batch_dates = []
    
    for (x, mask), y, dates in batch:
        batch_x_mask.append((x, mask))
        batch_y.append(y)
        batch_dates.append(dates)
    
    # Unzip the x and mask components
    xs, masks = zip(*batch_x_mask)
    
    # Find the maximum sequence length in the batch
    max_len = max([x.shape[0] for x in xs])
    
    # Initialize lists to store padded data
    padded_x = []
    padded_mask = []
    
    for x, mask in zip(xs, masks):
        # Get current sequence length
        curr_len = x.shape[0]
        
        # Convert numpy arrays to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        # Pad if needed
        if curr_len < max_len:
            padding = (0, 0, 0, 0, 0, max_len - curr_len)  # Pad first dimension (T)
            x_pad = F.pad(x, padding)
            mask_pad = mask  # Mask doesn't need temporal padding
        else:
            x_pad = x
            mask_pad = mask
        
        padded_x.append(x_pad.float())
        padded_mask.append(mask_pad.float())
    
    # Convert labels and dates to tensors if needed
    ys = [torch.tensor(y, dtype=torch.long) if isinstance(y, (int, np.integer)) else y.long() for y in batch_y]
    dates = [torch.tensor(d, dtype=torch.float32) if isinstance(d, (list, np.ndarray)) else d.float() for d in batch_dates]
    
    # Pad dates to match max_len
    padded_dates = []
    for d in dates:
        curr_len = d.shape[0]
        if curr_len < max_len:
            # Pad with the last date value
            padding = torch.full((max_len - curr_len,), d[-1].item(), dtype=torch.float32)
            d_pad = torch.cat([d, padding])
        else:
            d_pad = d
        padded_dates.append(d_pad)
    
    # Stack all tensors
    x_stack = torch.stack(padded_x)
    mask_stack = torch.stack(padded_mask)
    y_stack = torch.stack(ys)
    dates_stack = torch.stack(padded_dates)
    
    return (x_stack, mask_stack), y_stack, dates_stack


class PixelSetData(data.Dataset):
    def __init__(self, folder, labels, npixel, sub_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), sensor='HLS_S30', minimum_sampling=16, return_id=False):
        """
        Dataset class for HLS L30 data.
        
        Args:
            folder (str): path to the main folder of the dataset, formatted as indicated in the readme
            labels (str): name of the nomenclature to use in the labels.json file
            npixel (int): Number of sampled pixels in each parcel
            sub_classes (list): If provided, only the samples from the given list of classes are considered. 
            (Can be used to remove classes with too few samples)
            norm (tuple): (mean,std) tuple to use for normalization
            minimum_sampling (int) = minimum number of observation to sample
            - relevant where parcels have uneven number of observations.
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
        self.minimum_sampling = minimum_sampling
        self.extra_feature = extra_feature
        self.jitter = jitter  # (sigma, clip)
        self.sensor = sensor
        self.return_id = return_id

        # get parcel ids
        l = [f for f in os.listdir(self.data_folder) if f.endswith('.npy')]
        self.pid = [int(f.split('.')[0]) for f in l]
        self.pid = list(np.sort(self.pid))
        self.pid = list(map(str, self.pid))
        self.len = len(self.pid)

        # get Labels
        sub_indices = []
        num_classes = len(sub_classes)
        convert = dict((c, i) for i, c in enumerate(sub_classes))

        with open(os.path.join(folder, 'META', 'labels.json'), 'r') as file:
            d = json.loads(file.read())
            self.target = []
            for i, p in enumerate(self.pid):
                t = int(d[labels][p])
                if t in sub_classes:
                    sub_indices.append(i)
                    # Convert to 0-based index using position in sub_classes
                    self.target.append(sub_classes.index(t))
                else:
                    continue
                        
        # Update dataset to only include samples from selected classes
        self.pid = list(np.array(self.pid)[sub_indices])
        self.target = list(np.array(self.target)[sub_indices])
        self.len = len(sub_indices)
        
        # Store number of classes for model configuration
        self.num_classes = num_classes
            
        # get dates for positional encoding
        with open(os.path.join(folder, 'META', 'dates.json'), 'r') as file:
            d = json.loads(file.read())

        self.dates = [d[i] for i in self.pid]
        self.date_positions = [date_positions(i) for i in self.dates]
        
        # add extra features
        if self.extra_feature is not None:
            with open(os.path.join(self.meta_folder, '{}.json'.format(extra_feature)), 'r') as file:
                self.extra = json.loads(file.read())
            if isinstance(self.extra[list(self.extra.keys())[0]], (int, float)):
                for k in self.extra.keys():
                    self.extra[k] = [self.extra[k]]
            df = pd.DataFrame(self.extra).transpose()
            self.extra_m, self.extra_s = np.array(df.mean(axis=0)), np.array(df.std(axis=0))
            
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """Get a sample from the dataset.
        
        Returns:
            tuple: ((x, mask), y, dates)
                x: Tensor of shape (T, C, S) where T is sequence length, C is channels, S is number of pixels
                mask: Tensor of shape (S,) indicating valid pixels
                y: Integer class label (0-based)
                dates: Tensor of shape (T,) containing normalized dates
        """
        # Load raw data
        x0 = np.load(os.path.join(self.folder, 'DATA', '{}.npy'.format(self.pid[item])))
        y = self.target[item]
        item_date = self.date_positions[item]
        
        # Handle uneven number of observations for both HLS L30 and S30
        if self.minimum_sampling is not None:
            # If sequence is too long, randomly sample minimum_sampling timesteps
            if x0.shape[0] > self.minimum_sampling:
                indices = list(range(x0.shape[0]))
                random.shuffle(indices)
                indices = sorted(indices[:self.minimum_sampling])
                x0 = x0[indices, :, :]
                item_date = [item_date[i] for i in indices]
            # If sequence is too short, pad with zeros up to minimum_sampling
            elif x0.shape[0] < self.minimum_sampling:
                pad_size = self.minimum_sampling - x0.shape[0]
                x_pad = np.zeros((pad_size, x0.shape[1], x0.shape[2]))
                x0 = np.concatenate([x0, x_pad], axis=0)
                # Pad dates with the last date
                last_date = item_date[-1] if item_date else 0
                item_date.extend([last_date] * pad_size)
                # Pad dates with the last date repeated
                last_date = item_date[-1] if item_date else 0
                item_date.extend([last_date] * pad_size)

        if x0.shape[-1] > self.npixel:
            idx = np.random.choice(list(range(x0.shape[-1])), size=self.npixel, replace=False)
            x = x0[:, :, idx]
            mask = np.ones(self.npixel)

        elif x0.shape[-1] < self.npixel:
            if x0.shape[-1] == 0:
                x = np.zeros((*x0.shape[:2], self.npixel))
                mask = np.zeros(self.npixel)
                mask[0] = 1
            else:
                x = np.zeros((*x0.shape[:2], self.npixel))
                x[:, :, :x0.shape[-1]] = x0
                x[:, :, x0.shape[-1]:] = np.stack([x[:, :, 0] for _ in range(x0.shape[-1], x.shape[-1])], axis=-1)
                mask = np.array(
                    [1 for _ in range(x0.shape[-1])] + [0 for _ in range(x0.shape[-1], self.npixel)])
        else:
            x = x0
            mask = np.ones(self.npixel)

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
        x = x.astype('float32')

        if self.jitter is not None:
            sigma, clip = self.jitter
            x = x + np.clip(sigma * np.random.randn(*x.shape), -clip, clip)

        if self.extra_feature is not None:
            extra = np.array(self.extra[self.pid[item]])
            extra = (extra - self.extra_m) / (self.extra_s + 1e-8)

            if self.return_id:
                return (x, mask), extra, y, item
            else:
                return (x, mask), extra, y
        else:
            # Convert item_date to a tensor to ensure consistent types
            item_date_tensor = torch.tensor(item_date, dtype=torch.float32)
            
            # Convert to torch tensors with explicit float32 type
            x_tensor = torch.tensor(x, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            
            # Convert to tensors with explicit types
            x_tensor = torch.tensor(x, dtype=torch.float32)  # (T, C, S)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)  # (S,)
            y_tensor = torch.tensor(y, dtype=torch.long)  # Class label as long
            date_tensor = torch.tensor(item_date, dtype=torch.float32)  # (T,)
            
            if self.return_id:
                return (x_tensor, mask_tensor), y_tensor, date_tensor, item
            else:
                return (x_tensor, mask_tensor), y_tensor, date_tensor


class PixelSetData_preloaded(PixelSetData):
    """
    Wrapper class to load all the dataset to RAM at initialization (when the hardware permits it).
    Ensures all tensors are in float32 format for better performance.
    """
    def __init__(self, folder, labels, npixel, sub_classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), sensor=None, minimum_sampling=8, return_id=False):
        super(PixelSetData_preloaded, self).__init__(folder, labels, npixel, sub_classes, norm, extra_feature, jitter, sensor,
                                                     minimum_sampling, return_id)
        self.samples = []
        print('Loading samples to memory...')
        for item in range(len(self)):
            # Get sample from parent class
            (x, mask), y, dates = super(PixelSetData_preloaded, self).__getitem__(item)[:3]
            
            # Ensure all tensors are in correct format
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            elif x.dtype != torch.float32:
                x = x.to(torch.float32)
                
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32)
            elif mask.dtype != torch.float32:
                mask = mask.to(torch.float32)
                
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.long)
            elif y.dtype != torch.long:
                y = y.to(torch.long)
                
            if not isinstance(dates, torch.Tensor):
                dates = torch.tensor(dates, dtype=torch.float32)
            elif dates.dtype != torch.float32:
                dates = dates.to(torch.float32)
            
            # Store sample with correct types
            if self.return_id:
                self.samples.append(((x, mask), y, dates, item))
            else:
                self.samples.append(((x, mask), y, dates))
        print('Done!')

    def __getitem__(self, item):
        """Get a preloaded sample.
        
        Returns:
            tuple: ((x, mask), y, dates) or ((x, mask), y, dates, item) if return_id is True
                x: Tensor(float32) of shape (T, C, S)
                mask: Tensor(float32) of shape (S,)
                y: Tensor(long) class label
                dates: Tensor(float32) of shape (T,)
        """
        return self.samples[item]


def parse(date):
    year, month, day = date.split('-')
    return dt.datetime(int(year), int(month), int(day))


def interval_days(date1, date2):
    return abs((parse(date2) - parse(date1)).days)


def date_positions(dates):
    # Convert dates to positions (number of days since first acquisition)
    positions = []
    for d in dates:
        positions.append(interval_days(dates[0], d))
    return positions
