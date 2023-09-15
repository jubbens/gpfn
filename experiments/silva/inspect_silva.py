"""Inspect a sample drawn from a breeding prior"""

from loaders.simpop_prior import SimPopGenerator
from loaders.silva_prior import SilvaGenerator
from util.gblup import GBLUP
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import pearsonr

train_samples = 500
eval_samples = 100
num_families_max = 60

base_gen = SimPopGenerator(pop_max_size=num_families_max + 1, splitting=True, selection=False, verbose=True, use_cache=False)
breeding_prior = SilvaGenerator(base_population_generator=base_gen, num_train_samples=train_samples,
                                num_families_min=50, num_families_max=num_families_max,
                                family_size_min=20, family_size_max=25,
                                num_eval_samples=eval_samples, feature_selection=None)

x, y = breeding_prior[0]
print(x.shape)
print(np.unique(x))

# Viz

tsne_embs = TSNE(n_components=2).fit_transform(x)
print(tsne_embs.shape)

plt.figure()
plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], alpha=0.5, c=y)

plt.figure()
plt.hist(y[:train_samples], bins=100, alpha=0.5, label='training')
plt.hist(y[train_samples:], bins=100, alpha=0.5, label='testing')
plt.legend()

plt.figure()
plt.imshow(x)

# Test GBLUP
x_train = x[:train_samples]
x_test = x[train_samples:]
y_train = y[:train_samples]
y_test = y[train_samples:]

gb = GBLUP()
ebvs = gb.get_ebvs(x_train, x_test, y_train)
print('GBLUP r: {0}'.format(pearsonr(y_test, ebvs)[0]))

plt.figure()
plt.scatter(y_test, ebvs, alpha=0.5)

plt.show()
