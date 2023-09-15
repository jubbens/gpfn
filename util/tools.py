import numpy as np
import itertools
import torch
from joblib import load
from copy import deepcopy


def sample_interval(interval):
    if None in interval:
        return None
    elif isinstance(interval[0], int):
        return np.random.randint(interval[0], interval[1] + 1)
    else:
        return np.random.uniform(interval[0], interval[1])


def center_markers(m):
    m = m - m.min()
    m = m / m.max()
    return m - 0.5


def normalize(y):
    if len(y.shape) > 0:
        return (y - y.mean()) / y.std()
    else:
        return y


def get_random_mask(total_length, number_unmasked):
    """Get a random binary mask"""
    mask = np.array(([1] * number_unmasked) + ([0] * (total_length - number_unmasked)))
    np.random.shuffle(mask)

    return mask


def sort_into_families(inds):
    fams = {}

    for ind in inds:
        title = str(ind.sire) + str(ind.dam)

        if title in list(fams.keys()):
            fams[title].append(ind)
        else:
            fams[title] = [ind]

    return list(fams.values())


def flatten_list(l):
    return list(itertools.chain(*l))


def compress_genotypes(g):
    a = deepcopy(g)
    a = center_markers(a)
    a = (a + 0.5) * 2

    return a.astype(np.int8)


def decompress_genotypes(g):
    a = deepcopy(g)
    a = a.astype(np.float32)
    a = center_markers(a)

    return a


def load_from_cache_file(path):
    xs, ys = load(path)

    if xs.dtype == np.int8:
        xs = decompress_genotypes(xs)

    return xs, ys


def reencode_genotypes_as_major_and_minor(x):
    assert np.allclose(np.unique(x), np.array([-0.5, 0., 0.5])), 'trying to unphase genotypes but need them to be in [-0.5, 0., 0.5]'

    ret = np.zeros_like(x)
    allele_means = np.tile(np.mean(x, axis=0), (x.shape[0], 1))
    ret[(allele_means > 0.) & (x == 0.5)] = 0.5
    ret[(allele_means > 0.) & (x == -0.5)] = -0.5

    ret[(allele_means < 0.) & (x == -0.5)] = 0.5
    ret[(allele_means < 0.) & (x == 0.5)] = -0.5

    return ret


def save_model(model, path):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module, path)
    elif isinstance(model, torch.nn.Module):
        torch.save(model, path)
    else:
        raise Exception('Tried to save a model but it is of an unsupported class')


def load_model(path, device=None):
    if device in [None, torch.device('cpu')]:
        model = torch.load(path, map_location=torch.device('cpu'))
    else:
        # If we are targeting GPU, load all parameters onto GPU 0 for inference
        model = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))

    return model.module if isinstance(model, torch.nn.DataParallel) else model


def eval_sampler(min_training_samples, max_training_samples, pop_max_size, uniform=True):
    possibilities = np.arange(min_training_samples, max_training_samples)

    if uniform:
        probabilities = None
    else:
        probabilities = 1. / (pop_max_size - possibilities)
        probabilities = probabilities / np.sum(probabilities)

    return np.random.choice(possibilities, p=probabilities)
