"""Inspect a sample drawn from a breeding prior"""

from loaders.simpop_prior import SimPopGenerator
from loaders.recurrent_prior import RecurrentGenerator
from util.feature_selection import feature_reduce
from util.gblup import GBLUP
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

num_train = 990
num_eval = 500

base_gen = SimPopGenerator(pop_max_size=num_train, splitting=True, selection=False,
                           num_timesteps_min=50, num_timesteps_max=60,
                           verbose=True, use_cache=False)
breeding_prior = RecurrentGenerator(base_population_generator=base_gen,
                                    num_train_samples=num_train, num_eval_samples=num_eval,
                                    heritability_min=0.4, heritability_max=0.8,
                                    uo_ratio_min=None, uo_ratio_max=None,
                                    uo_mask_size_min=None, uo_mask_size_max=None,
                                    pregenerations_min=3, pregenerations_max=8,
                                    selection_intensity_min=0.01, selection_intensity_max=0.1,
                                    n_qtl_min=5, n_qtl_max=100,
                                    feature_selection=None,
                                    feature_max_size=None,
                                    use_cache=False, verbose=True)

x, y = breeding_prior[0]

train_x, train_y = x[:num_train], y[:num_train]
eval_x, eval_y = x[num_train:], y[num_train:]

print('Train:')
print(train_x.shape)
print(np.unique(train_x))
# Histogram of genotype values
print(np.histogram(train_x.flatten(), bins=3)[0] / train_x.flatten().shape[0])

print('Eval:')
print(eval_x.shape)
print(np.unique(eval_x))
# Histogram of genotype values
print(np.histogram(eval_x.flatten(), bins=3)[0] / eval_x.flatten().shape[0])

# Viz

plt.figure()
pca_embs = feature_reduce(train_x, train_y, 2, 'pca')
plt.scatter(pca_embs[:, 0], pca_embs[:, 1], alpha=0.5, c=train_y)

plt.figure()
tsne_embs = TSNE(n_components=2).fit_transform(train_x)
plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], alpha=0.5, c=train_y)

plt.figure()
plt.hist(train_y, bins=100, alpha=0.5, label='training')
plt.hist(eval_y, bins=100, alpha=0.5, label='testing')
plt.legend()

plt.figure()
plt.imshow(train_x)

# Test GBLUP

gb = GBLUP(h2=0.5, ploidy=2)
ebvs = gb.get_ebvs(train_x, eval_x, train_y)
print('GBLUP r: {0}'.format(pearsonr(eval_y, ebvs)[0]))

plt.figure()
plt.scatter(eval_y, ebvs, alpha=0.5)

plt.show()
