import pytest
from loaders.simpop_prior import SimPopGenerator
import numpy as np

pop_size = 100


@pytest.fixture(scope='module')
def generator():
    gen = SimPopGenerator(pop_max_size=pop_size, num_base_min=20, num_base_max=100,
                          num_train_samples=60, num_eval_samples=40,
                          feature_max_size=None, splitting=True, selection=False, verbose=False,
                          use_cache=False, feature_selection=None)

    return gen


@pytest.fixture(scope='module')
def sample(generator):
    return generator[0]


def test_sample_has_correct_shape(sample):
    assert sample[0].shape[0] == pop_size, 'generated population has the correct number of genotypes'
    assert sample[1].shape[0] == pop_size, 'generated population has the correct number of phenotypes'


def test_genotypes_centered(sample):
    x = sample[0]
    assert np.allclose(np.unique(x), np.array([-0.5, 0., 0.5])), 'genotype data should contain only [-0.5, 0., 0.5]'


def test_phenotypes_centered(sample):
    y = sample[1][:60]

    assert np.isclose(np.mean(y[:60]), 0., atol=1e-7), 'phenotypes are mean-centered'

