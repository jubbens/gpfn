from breeding_schemes.nam import NAM
from util.prior import Prior
from util.tools import sample_interval
from util.preprocessing import preprocess_population
import numpy as np
import timeit
from random import shuffle
from math import ceil


class NAMGenerator(Prior):
    def __init__(self, base_population_generator,
                 num_train_samples=960, num_eval_samples=40, feature_max_size=100,
                 num_families_min=25, num_families_max=50,
                 heritability_min=0.2, heritability_max=0.8,
                 dom_p_min=0., dom_p_max=0.25,
                 alpha_min=0.1, alpha_max=0.3,
                 beta_min=1., beta_max=9.5,
                 rr_min=0.5, rr_max=2.5,
                 uo_ratio_min=None, uo_ratio_max=None,
                 uo_mask_size_min=None, uo_mask_size_max=None,
                 n_qtl_min=5, n_qtl_max=100,
                 ssd_min=2, ssd_max=5,
                 feature_selection='pca',
                 use_cache=False, cache_dir='/tmp', asynchronous=False, verbose=False):
        super(NAMGenerator, self).__init__(use_cache=use_cache, cache_dir=cache_dir,
                                           asynchronous=asynchronous, verbose=verbose)

        self.base_generator = base_population_generator
        self.base_generator.set_preprocess(False)
        self.base_generator.set_unphase(False)
        # Don't mask any of the genotype in the base pop
        self.base_generator.uo_ratio_interval = [None, None]
        self.base_generator.uo_mask_size_interval = [None, None]

        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.feature_max_size = feature_max_size
        self.num_families_interval = [num_families_min, num_families_max]
        self.heritability_interval = [heritability_min, heritability_max]
        self.dom_p_interval = [dom_p_min, dom_p_max]
        self.alpha_interval = [alpha_min, alpha_max]
        self.beta_interval = [beta_min, beta_max]
        self.rr_interval = [rr_min, rr_max]
        self.n_qtl_interval = [n_qtl_min, n_qtl_max]
        self.uo_ratio_interval = [uo_ratio_min, uo_ratio_max]
        self.uo_mask_size_interval = [uo_mask_size_min, uo_mask_size_max]
        self.ssd_interval = [ssd_min, ssd_max]
        self.feature_selection = feature_selection

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, idx):
        # First we will rely on the superclass to try to load it from disk if applicable
        r = self.get_item_from_cache(idx)
        if r is not None:
            return r

        start_time = timeit.default_timer()

        # Dynamic family size depending on num families
        num_families = sample_interval(self.num_families_interval)
        total_pop_size = (self.num_train_samples + self.num_eval_samples)
        train_ratio = self.num_train_samples / total_pop_size
        num_train_families = ceil(num_families * train_ratio)
        num_eval_families = (num_families - num_train_families)
        family_size = max(ceil(self.num_train_samples / num_train_families),
                          ceil(self.num_eval_samples / num_eval_families))

        gen = NAM(num_families=num_families,
                  family_size=family_size,
                  train_families=num_train_families,
                  heritability=sample_interval(self.heritability_interval),
                  dom_p=sample_interval(self.dom_p_interval),
                  alpha=sample_interval(self.alpha_interval),
                  beta=sample_interval(self.beta_interval),
                  rr=sample_interval(self.rr_interval),
                  n_qtl=sample_interval(self.n_qtl_interval),
                  uo_ratio=sample_interval(self.uo_ratio_interval),
                  uo_mask_size=sample_interval(self.uo_mask_size_interval),
                  descent_generations=sample_interval(self.ssd_interval))

        f, _ = self.base_generator[idx]
        train_pop, eval_pop = gen.forward(founders=f)

        num_train_current = self.num_train_samples if self.eval_sched is None else self.eval_sched[idx]

        if len(train_pop) < num_train_current or len(eval_pop) < self.num_eval_samples:
            raise Exception('Generated a population with insufficient size')

        shuffle(train_pop)
        train_pop = train_pop[:num_train_current]
        shuffle(eval_pop)
        eval_pop = eval_pop[:self.num_eval_samples]

        train_x, train_y = np.stack([p.genotype for p in train_pop]), np.stack([p.phenotypes[0] for p in train_pop])
        eval_x, eval_y = np.stack([p.genotype for p in eval_pop]), np.stack([p.phenotypes[0] for p in eval_pop])

        if self.preprocess:
            train_x, train_y, eval_x, eval_y = preprocess_population(train_x, train_y, self.feature_max_size,
                                                                     self.feature_selection, eval_x=eval_x,
                                                                     eval_y=eval_y, do_unphase=self.unphase)

        xs = np.concatenate((train_x, eval_x), axis=0).astype(np.float32)
        ys = np.concatenate((train_y, eval_y)).astype(np.float32)

        if self.verbose:
            print('Generated a sample in {0:.2f} seconds'.format(timeit.default_timer() - start_time))

        # Use superclass to dump result to cache maybe
        self.write_item_to_cache(idx, xs, ys)

        return xs, ys
