""" Convolutional Neutral Network models to extract image features
"""

import os
import torch
from PIL import Image
from torch.utils import data
from torchvision.models import vgg16
from torchvision import transforms
import torch.nn as nn
import utilities as utils


class TruncatedModel(torch.nn.Module):
    """ Truncated model based on vgg16 pretrained model
        # TODO will generalize this to take any pretrained model
        with a profile
    """
    def __init__(self):
        super(TruncatedModel, self).__init__()
        model = vgg16(pretrained=True).eval()
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        self.model = model

    def forward(self, x, label=None):
        return self.model(x)


class ImageDataset(data.Dataset):
    """characterize a dataset for PyTorch that will be compatible
        built-in dataloader
    """
    def __init__(self, datadir, extension='.png', transform=None):
            """ initialize with list of filenames for dataset

        Args:
            datadir (str): path to images files
            extension (str): extensions specifying files to retrieve
            transform (torchvision transform object)
        """
        self.extension = extension
        self.datadir = datadir
        self.img_ids = list(utils.get_img_file_names(datadir, extension))
        if transform:
            self.transform = transform
        else:
            self.transform = \
                            transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        """ get total number of samples
        """
        return len(self.img_ids)

    def __getitem__(self, index):
        """ generates a single data sample

        Args:
            index(int): index of data to retrieve

        Returns:
             image
        """

        path = os.path.join(self.datadir, self.img_ids[index])
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': self.img_ids[index]}
