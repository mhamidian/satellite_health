""" Utility functions to streamline deep learning pipeline
"""

import torch
import pandas as pd
import os


FEATURE_FILE_LABEL_COL = 0
MAXIMUM_CHUNK_SIZE = 1000


def get_torch_device(verbose=False):
    """ Check device type to see if GPUs are available for training and inference
        Args:
            verbose (boolean): whether to display GPU parameters if present
        Return:
            (string) with torch device
    """

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if verbose:
            print('Cuda Device: ',
                  torch.cuda.device(torch.cuda.current_device()))
            print('Number of Cuda Devices: ',
                  torch.cuda.device_count())
            print('GPU Device: ',
                  torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")

    return device


def get_img_file_names(datadir, extension='.png'):
    """output the name of each file in the game path director
    Args:
        path(str): path in which to look for files
        estension (str): specifies files with a given extension

    Returns:
        list: A list of strings, each containing the name and extension for
        the file in the path
    """
    for file in os.listdir(datadir):
        if (os.path.isfile(os.path.join(datadir, file))) and \
           (file.endswith(extension)):

            yield file


def get_img_file_labels(filenames):
    """ use the standardized satellite image file names to extract
        a unique set of fips and image numbers

    Args:
        filenames (list of strings)

    Return:
        (list of dicts) containing original placename, fips id, and img number
    """

    labels = ['placename', 'fips_tract', 'img_n']

    return [{x: y for x, y in zip(labels, file.split('.')[0].split('_'))}
            for file in filenames]


def save_features_to_file(features, output_dir=None, output_filename=None):
    """ save feature vector numpy array and associated labels

    Args:
        features (list): label information, then elements of feature vectors
        output_dir (string)
        outputfile_names (string)

    Returns:
    """
    output_dir = output_dir or '../test_output'
    output_filename = output_filename or 'output.csv'
    # TODO check if file exists before appending
    with open(os.path.join(output_dir, output_filename), 'a') as f:
        df = pd.concat([pd.DataFrame(data=x) for x in features], axis=1)
        df.to_csv(f, header=False, index=False)


def get_features_by_tractid(path,
                            fips_ids=None,
                            label_column=FEATURE_FILE_LABEL_COL):
    """ Retrieve features from feature files using fips ids

    Args:
        path (string): location of feature file
        fips_ids (list of strings): if None return all
        label_columns (int): column in feature file that contains fips info

    Yields:
        (pandas Datafram) with feature vector and file name
    """

    df_labels = pd.read_csv(path, header=None, usecols=[label_column],
                            names=['labels'])

    if fips_ids:
        # find indices in data rows that have specified fips
        fips_row_index = df_labels[df_labels['labels']
                                   .apply(lambda x: x.split('_')[1])
                                   .isin(fips_ids)].index

        skiplist = set(range(0, len(df_labels))) - set(fips_row_index)
    else:
        skiplist = []

    for df in pd.read_csv(path, skiprows=skiplist,
                          chunksize=MAXIMUM_CHUNK_SIZE,
                          header=None):
        yield df


def average_feature_vectors(feature_file_path):
    """ averaging function for features vectors within the same fips
        NOTE: In the future this will be complicated function of
        geography, and other socioeconomic features

    Args:
        data_file_path (string): file contains feature vectors and their fips

    Returns:
        (Pd Dataframe) with averaged vectors and fips ids as columns
    """

    data = pd.DataFrame([])
    for df in get_features_by_tractid(feature_file_path):
        df['tract'], df['num_imgs'] = df.iloc[:, FEATURE_FILE_LABEL_COL] \
                                      .apply(lambda x: x.split('_')[1]), 1

        # TODO: generalize averaging function beyond simple average
        data = pd.concat([data, df], axis=0, sort=False) \
                 .groupby('tract').sum().reset_index()

    data = pd.concat([data['tract'],
                      data.iloc[:, 1:]
                     .apply(lambda row: row/row['num_imgs'], axis=1)], axis=1)\
             .drop('num_imgs', axis=1)

    data.columns = ['tract'] + ['x_{}'.format(x) for x in data.columns[1:]]
    return data


if __name__ == "__main__":
    # TEST
    path = "../test_output/output.csv"
    xx = ['25017353400']
    yy = ['25017354900']

    for x in get_features_by_tractid(path):
        print(x)
