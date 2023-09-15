from breeding_schemes.ril import RIL
from util.prior import Prior
from util.tools import sample_interval
from util.preprocessing import preprocess_population
import numpy as np
import timeit
from random import shuffle


class RILGenerator(Prior):
    def __init__(self, base_population_generator,
                 pop_max_size=1000, feature_max_size=100,
                 heritability_min=0.2, heritability_max=0.8,
                 dom_p_min=0., dom_p_max=0.25,
                 alpha_min=0.1, alpha_max=0.3,
                 beta_min=1., beta_max=9.5,
                 rr_min=0.5, rr_max=2.5,
                 uo_ratio_min=None, uo_ratio_max=None,
                 uo_mask_size_min=None, uo_mask_size_max=None,
                 n_qtl_min=5, n_qtl_max=100,
                 ssd_min=4, ssd_max=8,
                 feature_selection='pca',
                 use_cache=False, cache_dir='/tmp', asynchronous=False, verbose=False):
        super(RILGenerator, self).__init__(use_cache=use_cache, cache_dir=cache_dir,
                                           asynchronous=asynchronous, verbose=verbose)

        self.base_generator = base_population_generator
        self.base_generator.set_preprocess(False)
        self.base_generator.set_unphase(False)
        # Don't mask any of the genotype in the base pop
        self.base_generator.uo_ratio_interval = [None, None]
        self.base_generator.uo_mask_size_interval = [None, None]

        self.pop_max_size = pop_max_size
        self.feature_max_size = feature_max_size
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

        pop_max_size = sample_interval(self.pop_max_size) if isinstance(self.pop_max_size, list) else self.pop_max_size

        gen = RIL(num_children=pop_max_size,
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
        pop = gen.forward(founders=f)

        shuffle(pop)
        xs = np.stack([p.genotype for p in pop]).astype(np.float32)
        ys = np.stack([p.phenotypes[0] for p in pop]).astype(np.float32)

        # Truncate to pop max size
        if xs.shape[0] > pop_max_size:
            if self.verbose:
                print('Truncating population from {0} to {1}'.format(xs.shape[0], pop_max_size))
            xs = xs[:pop_max_size, :]
            ys = ys[:pop_max_size]
        elif xs.shape[0] < pop_max_size:
            raise Exception('Generated a population with insufficient size')

        if self.preprocess:
            xs, ys = preprocess_population(xs, ys, self.feature_max_size, self.feature_selection,
                                           do_unphase=self.unphase)

        if self.verbose:
            print('Generated a sample in {0:.2f} seconds'.format(timeit.default_timer() - start_time))

        # Use superclass to dump result to cache maybe
        self.write_item_to_cache(idx, xs, ys)

        return xs, ys
