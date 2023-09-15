"""Inspect a sample drawn from a breeding prior"""

from loaders.simpop_prior import SimPopGenerator
from loaders.nam_prior import NAMGenerator
from util.feature_selection import feature_reduce
from util.gblup import GBLUP
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

num_train = 990
num_eval = 500

base_gen = SimPopGenerator(pop_max_size=1000, num_base_min=200, num_base_max=1000, feature_selection=None,
                           verbose=True, use_cache=False)

breeding_prior = NAMGenerator(base_population_generator=base_gen,
                              num_train_samples=num_train, num_eval_samples=num_eval,
                              num_families_min=25, num_families_max=85,
                              feature_selection=None, feature_max_size=None,
                              use_cache=False, verbose=True)

x, y = breeding_prior[0]

train_x, train_y = x[:num_train], y[:num_train]
eval_x, eval_y = x[num_train:], y[num_train:]

all_x = np.concatenate((train_x, eval_x), axis=0)
all_y = np.concatenate((train_y, eval_y))
labels = [0] * train_y.shape[0] + [1] * eval_y.shape[0]

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
train_pca_embs, eval_pca_embs = feature_reduce(train_x, train_y, 100, 'pca', eval_x=eval_x)
all_pca_embs = np.concatenate((train_pca_embs, eval_pca_embs), axis=0)
plt.scatter(all_pca_embs[:, 0], all_pca_embs[:, 1], alpha=0.5, c=labels)

plt.figure()
tsne_embs = TSNE(n_components=2).fit_transform(all_pca_embs)
plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], alpha=0.5, c=labels)

plt.figure()
plt.hist(all_y, bins=100)

plt.figure()
plt.imshow(all_x)

plt.matshow(all_pca_embs)

# Test GBLUP
gb = GBLUP()
ebvs = gb.get_ebvs(train_x, eval_x, train_y)
print('GBLUP r: {0}'.format(pearsonr(eval_y, ebvs)[0]))

plt.figure()
plt.scatter(eval_y, ebvs, alpha=0.5)

plt.show()
