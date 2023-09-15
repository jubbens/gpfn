import pytest
from util.feature_selection import feature_reduce
from loaders.simpop_prior import SimPopGenerator
from copy import deepcopy
import numpy as np

pop_size = 100
target_feature_size = 50
methods = ['greedy', 'correlation', 'mi', 'regression', 'rp', 'g', 'pca', 'strided_pca', 'downsample']


@pytest.fixture(scope='module')
def generator():
    gen = SimPopGenerator(pop_max_size=pop_size, num_train_samples=60, num_eval_samples=40,
                          feature_max_size=None, splitting=True, selection=False, verbose=False,
                          use_cache=False, feature_selection=None)

    return gen


@pytest.fixture(scope='module')
def sample(generator):
    return generator[0]


def test_feature_selection_has_no_side_effects(sample):
    x, y = sample

    for method in methods:
        x_original = deepcopy(x)
        y_original = deepcopy(y)
        reduced = feature_reduce(x, y, target_feature_size, method=method)

        assert np.allclose(x, x_original), 'method {0} should not change the input genotype'.format(method)
        assert np.allclose(y, y_original), 'method {0} should not change the input phenotype'.format(method)


def test_feature_selection_returns_correct_shape(sample):
    x, y = sample

    for method in methods:
        reduced = feature_reduce(x, y, target_feature_size, method=method)

        assert reduced.shape[0] == x.shape[0], 'feature selection should not reduce the number of samples'
        assert reduced.shape[1] == target_feature_size, 'for {0}, number of features should be as dictated'.format(method)
