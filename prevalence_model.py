""" Set of models to predict prevalence """

import sklearn
from sklearn.linear_model import ElasticNet
import utilities as utils
import regression

FEATURE_FILE_LABEL_COL = 0
DISEASE_COL_POSTFIX = '_CrudePrev'


class PrevalenceModel():
    """ General Class of Models Used to Predict Disease Prevalence
    """
    pass


class BayesPrevalenceModel(PrevalenceModel):
    """ Makes predictions using Bayesian inference
    """
    pass


class ScikitPrevalenceModel(PrevalenceModel):
    """ Make predictions using scikit-learn regression models
        Uncertainty comes from bootstrapping
    """
    def __init__(self, diseases, feature_file_path,
                 fips_ids, n_bootstrap=1):
        """ initialize with list of filenames for dataset
            Create feature / disease prevalence dataset

        Args:
            disease (list of strings)
            feature_file_path (str): path to saved feature fectors and labels
            fips_ids (list): list of census tracts to extract disease data for
            n_bootstrap (int): number of iterations for bootstrapping
        """

        self.bootstrap = n_bootstrap
        self.feature_path = feature_file_path
        self.diseases = diseases
        self.fips_ids = fips_ids

        features_df = utils.average_feature_vectors(self.feature_path)
        self.dataset = regression.assemble_feature_prevalence_set(self.fips_ids,
                                                                  self.diseases,
                                                                  features_df)

    def create_regressors(self):

        return regression.create_regressors(self.dataset)

    def create_model(self):
        raise NotImplementedError()

    def train(self, verbose=False):
        X = self.create_regressors()
        self.models = {}
        for disease in self.diseases:
            y = self.dataset.loc[:, self.dataset.columns == disease+DISEASE_COL_POSTFIX]
            self.models[disease] = [self.create_model() for i in range(self.bootstrap)]

            if verbose:
                print(disease + ": ", end="")

            for model in self.models[disease]:
                if self.n_bootstrap == 1:
                    model.fit(X, y)
                else:
                    if verbose:
                        print("#", end="")
                    model.fit(*sklearn.utils.resample(X, y))

            if verbose:
                print()


class ElasticNetPrevalenceModel(ScikitPrevalenceModel):

    def create_model(self):
        return ElasticNet()


class StackedPrevalenceModel(ScikitPrevalenceModel):
    pass


if __name__ == '__main__':
    pass
