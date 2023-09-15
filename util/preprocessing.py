from util.tools import normalize, center_markers, reencode_genotypes_as_major_and_minor
from util.feature_selection import feature_reduce
import numpy as np


def preprocess_population(xs, ys, feature_max_size, feature_selection, eval_x=None, eval_y=None, do_unphase=False):
    if eval_y is not None:
        eval_y = eval_y - np.mean(ys)
        eval_y = eval_y / np.std(ys)

    xs = center_markers(xs)
    ys = normalize(ys)

    if eval_x is not None:
        eval_x = center_markers(eval_x)

    if do_unphase:
        xs = reencode_genotypes_as_major_and_minor(xs)

        if eval_x is not None:
            eval_x = reencode_genotypes_as_major_and_minor(eval_x)

    # Perform feature selection if applicable
    if feature_max_size is not None and xs.shape[1] < feature_max_size:
        raise Exception('Requested to reduce data to {0} dimensions, but there are only {1} markers.'.format(
            feature_max_size, xs.shape[1]))

    if feature_selection is not None and (feature_selection == 'g' or feature_max_size < xs.shape[1]):
        if eval_x is None:
            xs = feature_reduce(xs, ys, num_features=feature_max_size, method=feature_selection)
        else:
            xs, eval_x = feature_reduce(xs, ys, num_features=feature_max_size, method=feature_selection, eval_x=eval_x)

    if eval_x is None:
        return xs, ys
    else:
        return xs, ys, eval_x, eval_y
