"""Inspect a sample synthetic population"""

from loaders.simpop_prior import SimPopGenerator
from util.tools import get_random_mask
from util.gblup import GBLUP
from util.feature_selection import feature_reduce
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

gen = SimPopGenerator(pop_max_size=1000, num_base_min=500, num_base_max=1000,
                      splitting=True, selection=False,
                      num_timesteps_min=20, num_timesteps_max=60,
                      feature_selection=None, feature_max_size=None,
                      use_cache=False, verbose=True)

x, y = gen[0]

print(x.shape)
print(np.unique(x))
print(y.shape)
# Histogram of genotype values
print(np.histogram(x.flatten(), bins=3)[0] / x.flatten().shape[0])

# Viz

plt.figure()
plt.imshow(x)

plt.figure()
plt.hist(y, bins=100)

plt.figure()
pca_embs = feature_reduce(x, y, 2, 'pca')
plt.scatter(pca_embs[:, 0], pca_embs[:, 1], alpha=0.5, c=y)

plt.figure()
tsne_embs = TSNE(n_components=2).fit_transform(x)
plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], alpha=0.5, c=y)

G = feature_reduce(x, y, num_features=x.shape[0], method='g')
plt.matshow(G)

# Test GBLUP
mask = get_random_mask(x.shape[0], int(x.shape[0] * 0.8))
x_train = x[mask == 1]
x_test = x[mask == 0]
y_train = y[mask == 1]
y_test = y[mask == 0]

gb = GBLUP()
ebvs = gb.get_ebvs(x_train, x_test, y_train)
print('GBLUP r: {0}'.format(pearsonr(y_test, ebvs)[0]))

plt.figure()
plt.scatter(y_test, ebvs, alpha=0.5)

plt.show()
