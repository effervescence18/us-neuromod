import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TFUScapesDataset(Dataset):
    """
    PyTorch Dataset for TFUScapes ultrasound simulation data.
    """

    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        filepath = os.path.join(self.data_dir, filename)
        data = np.load(filepath)

        ct = torch.from_numpy(data['ct']).float()
        tr_coords = torch.from_numpy(data['tr_coords']).float()
        pmap = torch.from_numpy(data['pmap']).float()

        return ct, tr_coords, pmap


def load_split_lists(dataset_name="vinkle-srivastav/TFUScapes"):
    from datasets import load_dataset
    ds = load_dataset(dataset_name)
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        splits[split_name] = [item['text'] for item in ds[split_name]]
    return splits
