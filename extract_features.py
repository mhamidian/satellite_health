""" This module is responsible for image feature extraction using
pretrained NN """


import os
import torch
from torch.utils import data
from torchvision import transforms
import utilities as utils
import feature_models as fm


# temporary placeholder for image directory
# TODO put this in an .env or profile
parent_data_dir = '../xy-imagery/static-images'
image_provider_dir = 'google-maps'
output_dir = '../test_output'


def data_transformations(data_train=False, model_profile=None):
    """ Transform images to be optimal for CNN analysis

    Args:
        data_type (string): determines if incoming data is for training
        model_profile (dict): model-specific transformation parameters (TODO)

    Return:
        (torchvision transform objects)
    """
    # TODO implement method to read specific tranformation requirements for
    # pretrained CNNs based on model profiles

    # Image transformations
    if data_train:
        # Training data can use augmentation uses data augmentation
        data_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.ColorJitter(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
    # Evaluation data does not use augmentation
    else:
        data_transforms = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ])

    return data_transforms


def calculate_city_feature_vectors(state, city, model):
    """ forward pass images through model and extract feature vectors
    Args:
        state (string)
        city (string)
        model (class) defining Pytorch model with forward pass function

    Yields:
        (lists) of feature vector and associated image file name
    """

    dataloader_params = {'batch_size': 4, 'shuffle': False, 'num_workers': 4}
    device = utils.get_torch_device()

    path = os.path.join(parent_data_dir, image_provider_dir,
                        state, city, 'images')

    evaluation_dataset = fm.ImageDataset(path, transform=data_transformations())

    evaluation_set_generator = data.DataLoader(evaluation_dataset,
                                               **dataloader_params)

    with torch.set_grad_enabled(False):
        for i, batch in enumerate(evaluation_set_generator):
            # transfer to availble computation cores (CPU or GPU)
            print(i)
            batch['image'] = batch['image'].to(device)
            batch_features = model.forward(batch['image'])

            yield batch['label'], [x.numpy() for x in batch_features]


if __name__ == "__main__":
    # TEST
    state = 'MA'
    city = 'Cambridge'
    model = fm.TruncatedModel()

    for i, res in enumerate(calculate_city_feature_vectors(state, city,
                                                           model)):
        if i < 3:
            utils.save_features_to_file(res)
        else:
            break
