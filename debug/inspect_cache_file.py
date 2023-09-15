from joblib import load
import numpy as np
from util.tools import center_markers, normalize, get_random_mask
from util.gblup import GBLUP
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from os import path
from sys import argv

data_dir = argv[1]
idx = 0

eval_sched = load(path.join(data_dir, 'metadata.bin'))
eval_pos = eval_sched[idx]
file = path.join(data_dir, 'idx-{0}'.format(idx))

data = load(file)
x = center_markers(np.array(data[0]))
y = normalize(np.array(data[1]))
print(x.shape)
print(y.shape)
print(eval_pos)

if eval_pos is None:
    mask = get_random_mask(x.shape[0], int(float(x.shape[0]) * 0.8))
    train_x = x[mask == 1]
    eval_x = x[mask == 0]
    train_y = y[mask == 1]
    eval_y = y[mask == 0]
else:
    train_x, train_y = x[:eval_pos], y[:eval_pos]
    eval_x, eval_y = x[eval_pos:], y[eval_pos:]

# Histogram of genotype values
print(np.histogram(x.flatten(), bins=3)[0] / x.flatten().shape[0])

plt.matshow(x)

labels = [0] * train_y.shape[0] + [1] * eval_y.shape[0]

plt.figure()
# In the cache file, PCA has already been applied
pca_embs = np.concatenate((train_x, eval_x))
plt.scatter(pca_embs[:, 0], pca_embs[:, 1], alpha=0.5, c=labels)

plt.figure()
tsne_embs = TSNE(n_components=2).fit_transform(pca_embs)
plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], alpha=0.5, c=labels)

plt.figure()
plt.hist(y, bins=100)

# Test GBLUP
gb = GBLUP()
ebvs = gb.get_ebvs(train_x, eval_x, train_y)
print('GBLUP r: {0}'.format(pearsonr(eval_y, ebvs)[0]))

plt.figure()
plt.scatter(eval_y, ebvs, alpha=0.5)

plt.show()
