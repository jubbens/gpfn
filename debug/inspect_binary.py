from joblib import load
import numpy as np
from util.feature_selection import feature_reduce
from util.tools import normalize, get_random_mask
from util.gblup import GBLUP
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
from sys import argv

split_meta = False

file = argv[1]
eval_file = argv[2] if len(argv) > 2 else None

data = load(file)

if split_meta:
    x = np.concatenate([data[i][0] for i in range(len(data))])
    y = np.concatenate([data[i][1] for i in range(len(data))])
else:
    x = np.array(data[0])
    y = normalize(np.array(data[1]))

print(x.shape)
print(y.shape)

# Histogram of genotype values
print(np.histogram(x.flatten(), bins=3)[0] / x.flatten().shape[0])

plt.matshow(x)

G = feature_reduce(x, y, x.shape[0], method='g')
plt.matshow(G)

plt.figure()
pca_embs = feature_reduce(x, y, 2, 'pca')
plt.scatter(pca_embs[:, 0], pca_embs[:, 1], alpha=0.5, c=y)

plt.figure()
tsne_embs = TSNE(n_components=2).fit_transform(x)
plt.scatter(tsne_embs[:, 0], tsne_embs[:, 1], alpha=0.5, c=y)

plt.figure()
plt.hist(y, bins=100)

# Test GBLUP
if eval_file is None:
    mask = get_random_mask(x.shape[0], int(float(x.shape[0]) * 0.8))
    train_x = x[mask == 1]
    eval_x = x[mask == 0]
    train_y = y[mask == 1]
    eval_y = y[mask == 0]
else:
    train_x = x
    train_y = y

    data = load(eval_file)
    eval_x = np.array(data[0])
    eval_y = normalize(np.array(data[1]))

print(eval_x.shape)
print(eval_y.shape)

gb = GBLUP()
ebvs = gb.get_ebvs(train_x, eval_x, train_y)
print('GBLUP r: {0}'.format(pearsonr(eval_y, ebvs)[0]))

plt.figure()
plt.scatter(eval_y, ebvs, alpha=0.5)

plt.show()
