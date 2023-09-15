import pytest
from util.sbwrapper import Trial
from util.breedstrats import random_selection
import numpy as np

num_base = 100


@pytest.fixture(scope='module')
def experiment():
    experiment = Trial()
    experiment.generate_random_founders(nbase=num_base, nsnp=20000)
    experiment.define_traits(h2=[0.5], nqtl=10)

    return experiment


def test_return_type(experiment):
    assert isinstance(experiment, Trial), 'simulation returns a Trial object'


def test_make_founder_generation(experiment):
    assert experiment.get_number_of_generations() == 0, 'trial should start with zero generations'

    experiment.make_founder_generation()

    assert experiment.get_number_of_generations() == 1, 'making founders should add a generation'


def test_get_all_generations(experiment):
    assert isinstance(experiment.get_all_generations(), list), 'get_all_generations should return a list'


def test_get_all_individuals(experiment):
    assert isinstance(experiment.get_all_individuals(), list), 'get_all_individuals should return a list'


def test_correct_number_of_progeny(experiment):
    num_crosses = 10
    num_children = 3

    experiment.make_founder_generation()

    crosses = random_selection(experiment.get_latest_generation(), n=num_crosses, method='selfing')
    experiment.make_crosses(crosses, num_children=num_children)

    F1 = experiment.get_latest_generation()

    assert len(F1) == num_crosses * num_children, 'making crosses results in the expected number of progeny'


def test_crosses_do_not_change_previous_generation(experiment):
    num_crosses = 10
    num_children = 3

    experiment.make_founder_generation()
    founders = experiment.get_generation(0)

    crosses = random_selection(founders, n=num_crosses, method='selfing')
    experiment.make_crosses(crosses, num_children=num_children)

    founders_2 = experiment.get_generation(0)

    founders.sort(key=lambda l: l.id, reverse=True)
    founders_2.sort(key=lambda l: l.id, reverse=True)
    diffs = [founders[i].__dict__ == founders_2[i].__dict__ for i in range(len(founders))]

    assert np.all(diffs), 'making crosses does not change previous generation'


def test_insert_founders_returns_same(experiment):
    # Grab founders out of the existing experiment
    experiment.make_founder_generation()
    founders_x = np.stack([f.genotype for f in experiment.get_generation(0)])

    # Make a new experiment and stick them in
    trial = Trial()
    trial.insert_founders(founders_x, ploidy=2, nchrom=24)
    trial.define_traits(h2=[0.5], nqtl=1)
    trial.make_founder_generation()

    founders_2_x = np.stack([f.genotype for f in trial.get_generation(0)])

    assert founders_x.shape == founders_2_x.shape, \
        'individuals returned should have the same shape as individuals inserted'

    assert np.allclose(np.sum(founders_x, axis=0), np.sum(founders_2_x, axis=0)), \
        'individuals returned should have the same genotypes as individuals inserted'


def test_progeny_in_cross_order(experiment):
    num_crosses = 10
    num_children = 3

    experiment.make_founder_generation()
    founders = experiment.get_generation(0)

    crosses = random_selection(founders, n=num_crosses, method='selfing')
    experiment.make_crosses(crosses, num_children=num_children)

    F1s = experiment.get_latest_generation()

    # print(list(range(0, len(F1s), 3)))
    # print(list(range(27, 27+3)))
    # print(len(F1s))

    parents_match = [np.all([F1s[j].dam == crosses[i][0] and F1s[j].sire == crosses[i][1]
                             for j in range(i * num_children, (i * num_children) + num_children)])
                     for i in range(0, len(crosses))]

    assert np.all(parents_match), 'the order of progeny should match the order of the crosses'
