import os
import pandas as pd
import numpy as np
import torchvision.datasets
import pathlib

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: pathlib.Path, root_dir: pathlib.Path, transform=None, class_index=1) -> None:
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.class_index = class_index

    @property
    def file_extension(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0] + self.file_extension)
        image = Image.open(img_name)

        label = self.data.iloc[idx, self.class_index]

        if self.transform:
            image = self.transform(image)

        return image, label


class APTOS2019Dataset(CsvImageDataset):
    def __init__(self, csv_path: pathlib.Path, root_dir: pathlib.Path, transform=None) -> None:
        super().__init__(csv_path, root_dir, transform)
        self.targets = self.data.iloc[:, self.class_index].values.tolist() 

    @property
    def file_extension(self):
        return '.png'


def get_dataset(dataset_name: str, datasets_path: pathlib.Path, train=True):
    if dataset_name == 'APTOS2019':
        csv_path = datasets_path / 'APTOS2019'
        root_dir = datasets_path / 'APTOS2019'
        if train:
            csv_path = csv_path / 'train.csv'
            root_dir = root_dir / 'train_images'
        else:
            csv_path = csv_path / 'test.csv'
            root_dir = root_dir / 'test_images'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = APTOS2019Dataset(csv_path, root_dir, transform)
    elif dataset_name == 'GLAUCOMA':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(datasets_path / 'GLAUCOMA', transform=transform, is_valid_file=is_valid_file)
    elif dataset_name == 'JSIEC':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(datasets_path / 'JSIEC', transform=transform, is_valid_file=is_valid_file)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return dataset


def is_valid_file(path):
    return not pathlib.Path(path).name.startswith('.')
