from util.tools import load_from_cache_file, compress_genotypes
import numpy as np
from torch.utils.data import Dataset
from joblib import dump
import os
import time


class Prior(Dataset):
    def __init__(self, use_cache=False, cache_dir='/tmp', asynchronous=False, verbose=False):
        super().__init__()
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.asynchronous = asynchronous
        self.verbose = verbose
        self.delete_cached = False
        self.preprocess = True
        self.unphase = True
        self.eval_sched = None

    def set_preprocess(self, p):
        self.preprocess = p

    def set_unphase(self, u):
        self.unphase = u

    def set_delete_cached(self):
        self.delete_cached = True

    def set_eval_sched(self, s):
        self.eval_sched = s

    def get_item_from_cache(self, idx):
        cache_file = os.path.join(self.cache_dir, 'idx-{0}'.format(idx))

        if self.use_cache:
            if self.asynchronous:
                while not os.path.exists(cache_file):
                    time.sleep(1)

            if os.path.isfile(cache_file):
                if self.verbose:
                    print('Loading from cache file {0}'.format(cache_file))
                try:
                    ret = load_from_cache_file(cache_file)

                    if self.asynchronous and self.delete_cached:
                        if self.verbose:
                            print('Removing cache file {0}'.format(cache_file))

                        os.remove(cache_file)

                    return ret
                except:
                    print('--- Error loading from cache file, will re-process ---')

        return None

    def write_item_to_cache(self, idx, xs, ys):
        if self.use_cache:
            cache_file = os.path.join(self.cache_dir, 'idx-{0}'.format(idx))

            if self.verbose:
                print('Dumping to cache file {0}'.format(cache_file))
            if np.array_equal(np.unique(xs), np.array([-0.5, 0., 0.5])):
                # These appear to by raw SNPs we're saving, so convert them to int8 first
                dump((compress_genotypes(xs), ys), cache_file)
            else:
                dump((xs, ys), cache_file)
