""" Routines to perform regression with image features and
locale indicators """


import numpy as np
import db_access as db
import utilities as utils
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COL_PREFIX = 'x_'
DISEASE_COL_POSTFIX = '_CrudePrev'


def bin_records(x, w, m, labels):
    """Bins data into approximately evenly weighted bins

    Args:
        x (iterable): A list of inputs. NaN and None values are allowed.
        w (iterable): A list of nonnegative weights. The number of weights must
                      equal the number of inputs.
        m: A nonnegative number of bins, to group the inputs. m must be less
           than or equal to the number of inputs.
        labels: A list of bin names. The number of labels must equal m.

    Returns:
        np.array: A list of bin labels for each of the input values,
                  such that the weight of each bin is approximately equal.
    """
    # check preconditions
    assert len(x) == len(w) >= m == len(labels)
    orig_df = pd.DataFrame({'val': x, 'wt': w})
    # remove illegal values
    df = orig_df[np.isfinite(orig_df.val)]
    # shuffle and sort (shuffling will effectively dither values at boundary)
    df = df.sample(frac=1).sort_values('val')
    # equally spaced bin boundaries by weight
    wt_bin_bounds = np.linspace(0, np.sum(df.wt), m + 1)
    # assign bins according to cumulative sum of weights for each value
    # bin 0 is values with weight less than 0, so subtract 1 from all bin indices
    bin_indices = np.digitize(np.cumsum(df.wt), wt_bin_bounds, right=True) - 1
    # assign bin labels according to bin indices
    df = df.assign(bin_label=np.array(labels)[bin_indices])
    # re-include nan-values, sort into original order
    orig_df = orig_df.assign(bin_label=df.bin_label)
    return orig_df.bin_label.values


def train_test_split_by_bin(data, test_size, bin_column):
    """Split data into training and test sets while preserving bin proportions
    Args:
        data (dataframe): Dataframe of summary data
        test_size (float): Number that sets the proportion of data to keep as
        test set bin_columns: Column in data that labels the bins
    Returns:
        D_train (dataframe): training subset of data
        d_test (dataframe): test subset of data
    """
    D_train = pd.DataFrame([])
    d_test = pd.DataFrame([])

    for x in set(data[bin_column].values):
        subset_data = data[data[bin_column] == x]
        train, test = train_test_split(subset_data, test_size=test_size)
        D_train = D_train.append(train)
        d_test = d_test.append(test)

    return D_train.reset_index(drop=True), d_test.reset_index(drop=True)


def assemble_feature_prevalence_set(region_fips, diseases, avg_features):
    """ Assemble average feature vectors and disease prevalences from database

    Args:
        regions_fips (list)
        disease (list): list disease to retrieve information for
        avg_features (pd.DataFrame): contains averaged feature vectors and
                                     fips ids

    Returns:
        (Pd Dataframe)
    """
    fips_data = pd.DataFrame(columns=['fips_tract'])
    for disease in diseases:
        data = db.get_fips_data(region_fips, disease)
        fips_data = fips_data.merge(data,
                                    left_on='fips_tract',
                                    right_on='fips_tract', how='outer')

        fips_data.drop(['stateabbr', 'placename', 'fips_place_tract'], axis=1, inplace=True) 

    fips_data['fips_tract'] = fips_data['fips_tract'].astype(str)

    return fips_data.merge(avg_features, left_on='fips_tract', right_on='tract')


def create_regressors(dataset):
    """ Split assembled feature vector disease prevalance dataset
        NOTE: This function will be built with categorical variables

    Args:
        dataset (pd DataFrame)
    """
    X = dataset.filter(regex='^'+FEATURE_COL_PREFIX, axis=1)

    return X

if __name__ == "__main__":
    # TEST
    path = "../test_output/output_Cambridge.csv"
    city = 'Cambridge'
    state = 'MA'
    # x = average_feature_vectors(path)
    features_df = utils.average_feature_vectors(path)
    region_fips = list(db.get_region_fips(state, city)['fips_place_tract'])
    x = assemble_feature_prevalence_set(region_fips, ['ARTHRITIS', 'BINGE'], features_df)
    print(list(x.columns[0:50]))
