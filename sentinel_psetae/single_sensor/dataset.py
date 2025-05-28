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


class PixelSetData(data.Dataset):
    def __init__(self, folder, labels, npixel, sub_classes=None, norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), sensor=None, minimum_sampling=27, return_id=False, positions=None):
        """
        
        Args:
            folder (str): path to the main folder of the dataset, formatted as indicated in the readme
            labels (str): name of the nomenclature to use in the labels.json file
            npixel (int): Number of sampled pixels in each parcel
            sub_classes (list): If provided, only the samples from the given list of classes are considered. 
            (Can be used to remove classes with too few samples)
            norm (tuple): (mean,std) tuple to use for normalization
            minimum_sampling (int) = minimum number of observation to sample for Sentinel-2
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
        self.jitter = jitter  # (sigma , clip )
        self.sensor = sensor
        self.return_id = return_id
        self.positions = positions


        # get parcel ids
        l = [f for f in os.listdir(self.data_folder) if f.endswith('.npy')]
        self.pid = [int(f.split('.')[0]) for f in l]
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

            
        # get dates for positional encoding
        with open(os.path.join(folder, 'META', 'dates.json'), 'r') as file:
            d = json.loads(file.read())

        self.dates = [d[i] for i in self.pid]
        self.date_positions = [date_positions(i) for i in self.dates]
        

        # add extra features 
        if self.extra_feature is not None:
            with open(os.path.join(self.meta_folder, '{}.json'.format(extra_feature)), 'r') as file:
                self.extra = json.loads(file.read())

            if isinstance(self.extra[list(self.extra.keys())[0]], int):
                for k in self.extra.keys():
                    self.extra[k] = [self.extra[k]]
            df = pd.DataFrame(self.extra).transpose()
            self.extra_m, self.extra_s = np.array(df.mean(axis=0)), np.array(df.std(axis=0))
            

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        
        x0 = np.load(os.path.join(self.folder, 'DATA', '{}.npy'.format(self.pid[item])))
        y = self.target[item]
        item_date = self.date_positions[item]
        

        # Handle sequence length
        if self.sensor == 'S2' and self.minimum_sampling is not None:
            # For S2: randomly sample minimum_sampling dates
            if x0.shape[0] > self.minimum_sampling:
                indices = list(range(x0.shape[0]))
                random.shuffle(indices)
                indices = sorted(indices[:self.minimum_sampling])
                x0 = x0[indices, :,:]
                item_date = [item_date[i] for i in indices]
        
        # For both S1 and S2: use positional indices if positions=='order'
        if self.positions == 'order':
            # Use 0-based sequence positions, these will be mapped to embeddings in range(lms)
            item_date = list(range(x0.shape[0]))
            

        if x0.shape[-1] > self.npixel:
            # Sample random pixels
            idx = np.random.choice(x0.shape[-1], size=self.npixel, replace=False)
            x = x0[:, :, idx]
            mask = np.ones(self.npixel)
        elif x0.shape[-1] < self.npixel:
            # Pad with zeros if we have fewer pixels than needed
            x = np.zeros((*x0.shape[:2], self.npixel))
            mask = np.zeros(self.npixel)
            if x0.shape[-1] == 0:
                # If no pixels, set first mask element to 1
                mask[0] = 1
            else:
                # Copy existing data and set corresponding mask
                x[:, :, :x0.shape[-1]] = x0
                mask[:x0.shape[-1]] = 1
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
        x = x.astype('float')

        if self.jitter is not None:
            sigma, clip = self.jitter
            x = x + np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)

        data = (Tensor(x), Tensor(mask))

        if self.extra_feature is not None:
            
            ef = (self.extra[str(self.pid[item])] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()

            ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)

        if self.return_id:
            return data, torch.from_numpy(np.array(y, dtype=int)), Tensor(item_date), self.pid[item] 

        else:
            return data, torch.from_numpy(np.array(y, dtype=int)), Tensor(item_date)



class PixelSetData_preloaded(PixelSetData):
    """ Wrapper class to load all the dataset to RAM at initialization (when the hardware permits it).
    """
    def __init__(self, folder, labels, npixel, sub_classes=None, norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), sensor=None, minimum_sampling=27, return_id=False, positions=None):
        super(PixelSetData_preloaded, self).__init__(folder, labels, npixel, sub_classes, norm, extra_feature, jitter, sensor,
                                                     minimum_sampling, return_id, positions)
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
    for d in dates:
        pos.append(interval_days(d, dates[0]))
    return pos

def custom_collate(batch):
    """Custom collate function for S1 data.
    Each sample in batch is ((x, mask), y, dates) where:
    - x: (T, 2, S) tensor where T is variable sequence length
    - mask: (S,) tensor
    - y: class label
    - dates: (T,) tensor of positional encodings
    """
    # Split batch into components
    xs = []
    masks = []
    batch_y = []
    batch_dates = []
    
    for i, ((x, mask), y, dates) in enumerate(batch):
        # Convert numpy arrays to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if isinstance(dates, (list, np.ndarray)):
            dates = torch.tensor(dates, dtype=torch.float32)
        
        # Handle case where x is a tuple (x, mask)
        if isinstance(x, tuple):
            x, mask = x
        
        # Convert to torch tensors if numpy arrays
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        xs.append(x.float())
        masks.append(mask.float())
        batch_y.append(y)
        batch_dates.append(dates)
    
    # Get max sequence length in this batch
    sequence_lengths = [x.shape[0] for x in xs]
    max_len = max(sequence_lengths)
    
    # Pad sequences to max_len
    padded_x = []
    padded_dates = []
    
    for i, (x, dates) in enumerate(zip(xs, batch_dates)):
        # Get current sequence length
        curr_len = x.shape[0]
        
        # Pad if needed
        if curr_len < max_len:
            padding = (0, 0, 0, 0, 0, max_len - curr_len)  # Pad first dimension (T)
            x_pad = F.pad(x, padding)
            # Pad dates with last valid date value
            last_date = dates[-1] if dates.numel() > 0 else torch.tensor(0.)
            date_padding = last_date.repeat(max_len - curr_len)
            dates = torch.cat([dates, date_padding])
        else:
            x_pad = x
        
        padded_x.append(x_pad.float())
        padded_dates.append(dates.float())
    
    # Stack all tensors
    x_stack = torch.stack(padded_x)  # (B, T, 2, S)
    
    # Expand masks to match sequence length
    expanded_masks = [m.unsqueeze(0).expand(max_len, -1).float() for m in masks]  # (T, S) for each mask
    mask_stack = torch.stack(expanded_masks)  # (B, T, S)
    
    # Stack labels and dates
    y_stack = torch.stack([y.clone().detach().to(torch.int64) if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.int64) for y in batch_y])  # (B,)
    dates_stack = torch.stack(padded_dates)  # (B, T)
    return (x_stack, mask_stack), y_stack, dates_stack
