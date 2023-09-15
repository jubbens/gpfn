import pytest
from loaders.simpop_prior import SimPopGenerator
from loaders.silva_prior import SilvaGenerator
import numpy as np

base_pop_size = 100
family_size = 5
num_families = 10
num_train, num_eval = 20, 20


@pytest.fixture(scope='module')
def generator():
    base_gen = SimPopGenerator(pop_max_size=base_pop_size, num_base_min=20, num_base_max=100,
                               num_train_samples=60, num_eval_samples=40,
                               splitting=True, selection=False, verbose=False, use_cache=False)
    breeding_prior = SilvaGenerator(base_population_generator=base_gen,
                                    num_train_samples=num_train, num_eval_samples=num_eval,
                                    num_families_min=num_families, num_families_max=num_families,
                                    family_size_min=family_size, family_size_max=family_size,
                                    feature_selection=None)

    return breeding_prior


@pytest.fixture(scope='module')
def sample(generator):
    return generator[0]


def test_sample_has_correct_shape(sample):
    assert sample[0].shape[0] == num_train + num_eval, 'generated population has the correct number of genotypes'
    assert sample[1].shape[0] == num_train + num_eval, 'generated population has the correct number of phenotypes'


def test_genotypes_centered(sample):
    x = sample[0]
    assert np.allclose(np.unique(x), np.array([-0.5, 0., 0.5])), 'genotype data should contain only [-0.5, 0., 0.5]'


def test_phenotypes_centered(sample):
    train_y = sample[1][:num_train]

    assert np.isclose(np.mean(train_y), 0., atol=1e-7), 'phenotypes are mean-centered'
