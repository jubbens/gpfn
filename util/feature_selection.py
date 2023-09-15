import numpy as np
from util.SeqBreed import selection as sel
from sklearn.random_projection import SparseRandomProjection
from sklearn.feature_selection import mutual_info_regression, r_regression, SequentialFeatureSelector, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import warnings


def feature_reduce(x, y, num_features, method, eval_x=None):
    if method in [None, 'none']:
        if eval_x is None:
            return x
        else:
            return x, eval_x
    elif method == 'mi':
        mus = mutual_info_regression(x, y)
        mask = np.zeros_like(mus)
        mask[(-mus).argsort()[:num_features]] = 1

        if eval_x is None:
            return x[:, mask.astype(bool)]
        else:
            return x[:, mask.astype(bool)], eval_x[:, mask.astype(bool)]
    elif method == 'correlation':
        # If any of the snps have no minor alleles this will throw a warning
        # Catch the warning and zero out the corresponding entries so they aren't selected
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mus = np.abs(r_regression(x, y))

        mus[mus == np.inf] = 0.
        mask = np.zeros_like(mus)
        mask[(-mus).argsort()[:num_features]] = 1
        if eval_x is None:
            return x[:, mask.astype(bool)]
        else:
            return x[:, mask.astype(bool)], eval_x[:, mask.astype(bool)]
    elif method == 'sequential':
        dt = DecisionTreeRegressor()
        sfs = SequentialFeatureSelector(estimator=dt, n_features_to_select=num_features, n_jobs=1)
        sfs.fit(x, y)

        if eval_x is None:
            return x[:, sfs.get_support()]
        else:
            return x[:, sfs.get_support()], eval_x[:, sfs.get_support()]
    elif method == 'regression':
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            r = ElasticNet(fit_intercept=False, alpha=0.015).fit(x, y)

        sorted = (-np.abs(r.coef_)).argsort()

        if eval_x is None:
            return x[:, sorted[:num_features]]
        else:
            return x[:, sorted[:num_features]], eval_x[:, sorted[:num_features]]

    elif method == 'rp':
        srp = SparseRandomProjection(n_components=num_features)
        if eval_x is None:
            return srp.fit_transform(x)
        else:
            srp.fit(x)
            return srp.transform(x), srp.transform(eval_x)
    elif method == 'pca':
        with threadpool_limits(limits=1):
            pc = PCA(n_components=num_features)
            if eval_x is None:
                return pc.fit_transform(x)
            else:
                pc.fit(x)
                return pc.transform(x), pc.transform(eval_x)
    elif method == 'strided_pca':
        num_chunks = 20
        window_size = x.shape[1] // num_chunks
        features_per_chunk = num_features // num_chunks
        out = np.zeros((x.shape[0], num_features))

        if eval_x is None:
            for i in range(num_chunks):
                window_start, window_stop = (i * window_size), ((i + 1) * window_size)
                pc = PCA(n_components=features_per_chunk)
                pcs = pc.fit_transform(x[:, window_start:window_stop])
                out[:, i * features_per_chunk:(i + 1) * features_per_chunk] = pcs

            return out
        else:
            eval_out = np.zeros((eval_x.shape[0], num_features))

            for i in range(num_chunks):
                window_start, window_stop = (i * window_size), ((i + 1) * window_size)
                pc = PCA(n_components=features_per_chunk).fit(x[:, window_start:window_stop])
                out[:, i * features_per_chunk:(i + 1) * features_per_chunk] = pc.transform(x[:, window_start:window_stop])
                eval_out[:, i * features_per_chunk:(i + 1) * features_per_chunk] = pc.transform(eval_x[:, window_start:window_stop])

            return out, eval_out

    elif method == 'g':
        x_orig_size = x.shape

        if x.shape[0] < num_features:
            needed = num_features - x.shape[0]
            x = np.concatenate((x, np.tile(np.zeros(x[0].shape), (needed, 1))), axis=0)

        if eval_x is None:
            g = sel.doGRM(np.transpose(x))
            if g.shape[1] > num_features:
                g = g[:, :num_features]

            return g[:x_orig_size[0]]
        else:
            x2 = np.concatenate((x, eval_x))
            g = sel.doGRM(np.transpose(x2))
            if g.shape[1] > num_features:
                g = g[:, :num_features]

            return g[:x_orig_size[0]], g[-eval_x.shape[0]:]
    elif method == 'composite':
        if eval_x is None:
            g = feature_reduce(x, y, None, 'g')
            s = feature_reduce(x, y, num_features, 'mi')
            return np.concatenate((g, s), axis=1)
        else:
            g, g_eval = feature_reduce(x, y, None, 'g', eval_x)
            s, s_eval = feature_reduce(x, y, num_features, 'correlation', eval_x)
            return np.concatenate((g, s), axis=1), np.concatenate((g_eval, s_eval), axis=1)
    elif method == 'downsample':
        def downsample(a):
            a = a + 0.5
            window_size = a.shape[1] // num_features
            out = np.zeros((a.shape[0], num_features))

            for i in range(num_features):
                out[:, i] = np.mean(a[:, (i * window_size):((i + 1) * window_size)], axis=1)

            return out

        if eval_x is None:
            return downsample(x)
        else:
            return downsample(x), downsample(eval_x)
    elif method == 'greedy':
        data = x if eval_x is None else np.concatenate((x, eval_x), axis=0)
        C = np.cov(data.T)

        # First grab the feature with the highest variance
        max_var = np.argmax(np.diagonal(C))
        selected_idxs = [max_var]
        cum_cov = C[max_var]
        cum_cov[max_var] = np.inf

        # Mask out monomorphic indices
        mafs = np.apply_along_axis(lambda a: np.histogram(a, bins=3)[0] / len(a), 0, data)
        cum_cov[np.where(np.any(mafs > 0.85, axis=0))] = np.inf

        possible_valid = np.count_nonzero(np.where(cum_cov < np.inf))

        if possible_valid < num_features:
            raise ValueError('Requested {0} features but only {1} markers with MAF > 0.95'.format(num_features, possible_valid))

        for i in range(1, num_features):
            selected = np.argmin(cum_cov)
            selected_idxs.append(selected)
            cum_cov = np.maximum(cum_cov, C[selected])
            cum_cov[selected] = np.inf

        if eval_x is None:
            return x[:, selected_idxs]
        else:
            return x[:, selected_idxs], eval_x[:, selected_idxs]
    else:
        raise UserWarning('Unknown feature selection strategy {0}'.format(method))
