import pytest
from util.sbwrapper import Trial
from util.breedstrats import get_crosses

nbase = 100


@pytest.fixture(scope='module')
def generation():
    experiment = Trial()
    experiment.generate_random_founders(nbase=nbase, nsnp=20000)
    experiment.define_traits(h2=[0.5], nqtl=10)
    experiment.make_founder_generation()

    return experiment.get_generation(0)


def test_get_crosses_selfing(generation):
    crosses = get_crosses(generation, method='selfing')

    assert len(crosses) == nbase, 'selfing should return the number of individuals'


def test_get_crosses_common(generation):
    crosses = get_crosses(generation, method='common')

    assert len(crosses) == nbase - 1, 'common crossing should return the number of individuals minus one'


def test_get_crosses_pairs(generation):
    crosses = get_crosses(generation, method='pairs')

    assert len(crosses) == nbase / 2, 'pairs should return half the number of individuals'
