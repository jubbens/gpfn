import pytest
from loaders.simpop_prior import SimPopGenerator
from loaders.ril_prior import RILGenerator
import numpy as np

base_pop_size = 100
pop_max = 20


@pytest.fixture(scope='module')
def generator():
    base_gen = SimPopGenerator(pop_max_size=base_pop_size, num_base_min=20, num_base_max=100,
                               num_train_samples=60, num_eval_samples=40,
                               splitting=True, selection=False, verbose=False,
                               use_cache=False)
    breeding_prior = RILGenerator(base_population_generator=base_gen,
                                  pop_max_size=pop_max,
                                  feature_selection=None)

    return breeding_prior


@pytest.fixture(scope='module')
def sample(generator):
    return generator[0]


def test_sample_has_correct_shape(sample):
    assert sample[0].shape[0] == pop_max, 'generated population has the correct number of genotypes'
    assert sample[1].shape[0] == pop_max, 'generated population has the correct number of phenotypes'


def test_genotypes_centered(sample):
    x = sample[0]
    assert np.allclose(np.unique(x), np.array([-0.5, 0., 0.5])), 'genotype data should contain only [-0.5, 0., 0.5]'


def test_phenotypes_centered(sample):
    y = sample[1]

    assert np.isclose(np.mean(y), 0., atol=1e-7), 'phenotypes are mean-centered'
