from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import numpy as np
import imageio
import torch
import os


def extract_bayer_channels(raw):
    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]
    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)
    return RAW_norm

class LoadData(Dataset):
    def __init__(self, dataset_dir, dataset_size, eval=False, test=False):
        self.raw_dir = os.path.join(dataset_dir, 'mediatek_raw')
        self.dslr_dir = os.path.join(dataset_dir, 'fujifilm')
        self.dataset_size = dataset_size
        self.eval = eval
        self.test = test

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.eval:
            raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

            dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx) + '.png'))
            dslr_image = np.asarray(dslr_image)
            dslr_image = np.float32(dslr_image) / 255.0
            dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        elif self.test:
            raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx + 20000) + '.png')))
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

            dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx + 20000) + '.png'))
            dslr_image = np.asarray(dslr_image)
            dslr_image = np.float32(dslr_image) / 255.0
            dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        else:
            raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx + 4000) + '.png')))
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

            dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx + 4000) + '.png'))
            dslr_image = np.asarray(dslr_image)
            dslr_image = np.float32(dslr_image) / 255.0
            dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        return raw_image, dslr_image

class LoadVisualData(Dataset):
    def __init__(self, dataset_dir, dataset_size, full_resolution=False):
        if full_resolution:
            self.raw_dir = os.path.join(dataset_dir, 'full_resolution')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'mediatek_raw')
            self.dataset_size = dataset_size
        self.full_resolution = full_resolution

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.full_resolution:
            raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx) + '.png')))
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
        else:
            raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, str(idx + 3880) + '.png')))
            raw_image = extract_bayer_channels(raw_image)
            raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))
        return raw_image

if __name__ == '__main__':
    dataset_dir = '../dataset/'
    TRAIN_SIZE = 16000
    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, eval=False, test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    mean_b, mean_g, mean_r = 0, 0, 0
    train_iter = iter(train_loader)
    for i in tqdm(range(len(train_iter))):
        x, y = next(train_iter)
        mean_r += torch.mean(y[0, 0, :, :])
        mean_g += torch.mean(y[0, 1, :, :])
        mean_b += torch.mean(y[0, 2, :, :])
    print(mean_r * 255 / len(train_iter), mean_g * 255 / len(train_iter), mean_b * 255 / len(train_iter))