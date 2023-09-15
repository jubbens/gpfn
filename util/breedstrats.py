from itertools import combinations
import numpy as np
import random
from copy import deepcopy


def get_crosses(selected, method):
    selected = deepcopy(selected)

    if method == 'selfing':
        return [(x.id, x.id) for x in selected]
    elif method == 'exhaustive':
        return list(combinations([x.id for x in selected], 2))
    elif method == 'pairs':
        random.shuffle(selected)
        p = len(selected) if len(selected) % 2 == 0 else len(selected) - 1
        return [[selected[i].id, selected[i + 1].id] for i in np.arange(0, p, 2)]
    elif method == 'common':
        P0 = selected.pop()
        return [(P0.id, x.id) for x in selected]
    else:
        raise NotImplementedError('unknown crossing strategy: {0}'.format(method))


def phenotypic_selection(pop, n, trait_idx=0, method='pairs', negative=False):
    """Select the n individuals with the highest phenotypic value"""
    pop.sort(key=lambda x: x.phenotypes[trait_idx], reverse=(not negative))
    selected = pop[:n]
    crosses = get_crosses(selected, method=method)

    return crosses


def random_selection(pop, n, method='pairs'):
    """Select random individuals"""
    random.shuffle(pop)
    selected = pop[:n]
    crosses = get_crosses(selected, method=method)

    return crosses
