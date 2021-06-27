from torch.utils.data import Dataset
import json
from .data_preprocess import Attribute_builder
import torch
from pathlib import Path


class BrbDataset(Dataset):
    def __init__(self, annotations_file: Path,
                 attribute_builder: Attribute_builder):
        self.attribute_builder = attribute_builder
        self.annotation_data = self._read_data(annotations_file)

    def __len__(self):
        return len(self.annotation_data)

    def _read_data(self, annotations_file):
        annotation_data = []
        with open(annotations_file, 'r') as f:
            for line in f:
                data_dict = json.loads(line.strip())
                input_dict = data_dict['feature']
                label = data_dict['label']
                input_x = self.attribute_builder.transform(input_dict)
                annotation_data.append((torch.Tensor(input_x), label))
        return annotation_data

    def __getitem__(self, idx):
        input_x, label = self.annotation_data[idx]
        return input_x, label
