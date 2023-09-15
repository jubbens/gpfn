"""Inspect a sample drawn from a breeding prior"""

from loaders.simpop_prior import SimPopGenerator
from loaders.ril_prior import RILGenerator
from util.feature_selection import feature_reduce
from util.gblup import GBLUP
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

num_train = 800

base_gen = SimPopGenerator(pop_max_size=[200, 500], splitting=True, selection=False,
                           num_timesteps_min=20, num_timesteps_max=60,
                           verbose=True, use_cache=False)
breeding_prior = RILGenerator(base_population_generator=base_gen,
                              pop_max_size=1000,
                              feature_selection=None,
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

G, eval_G = feature_reduce(train_x, train_y, num_features=train_x.shape[0], method='g', eval_x=eval_x)
plt.matshow(np.concatenate((G, eval_G)))

plt.figure()
plt.hist(train_y, bins=100)

plt.figure()
plt.imshow(train_x)

# Test GBLUP

gb = GBLUP()
ebvs = gb.get_ebvs(train_x, eval_x, train_y)
print('GBLUP r: {0}'.format(pearsonr(eval_y, ebvs)[0]))

plt.figure()
plt.scatter(eval_y, ebvs, alpha=0.5)

plt.show()
