import torch.utils.data as Data
from PIL import Image


class Dataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_paths, labels, transform):
        'Initialization'
        self.labels = labels
        self.list_paths = list_paths
        self.transform= transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path = self.list_paths[index]

        # Load data and get label
        x = Image.open(path)
        X = self.transform(x)
        y = self.labels[path]

        return X, y
