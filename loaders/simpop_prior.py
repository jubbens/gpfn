from util.prior import Prior
from util.simpop import generate_random_population
from util.tools import sample_interval
from util.preprocessing import preprocess_population
import numpy as np
import timeit
from random import shuffle


class SimPopGenerator(Prior):
    def __init__(self, num_samples=100000, pop_max_size=1000,
                 num_train_samples=None, num_eval_samples=None,
                 num_base_min=500, num_base_max=1000,
                 feature_max_size=100, feature_selection='pca',
                 splitting=True, selection=False,
                 num_timesteps_min=20, num_timesteps_max=50,
                 p_split_min=0.1, p_split_max=0.3,
                 pop_snps_min=7000, pop_snps_max=36000,
                 heritability_min=0.2, heritability_max=0.8,
                 dom_p_min=0., dom_p_max=0.25,
                 alpha_min=0.1, alpha_max=0.3,
                 beta_min=1., beta_max=9.5,
                 rr_min=0.5, rr_max=2.5,
                 uo_ratio_min=None, uo_ratio_max=None,
                 uo_mask_size_min=None, uo_mask_size_max=None,
                 n_qtl_min=5, n_qtl_max=100,
                 use_cache=False, cache_dir='/tmp', asynchronous=False, verbose=False):
        super(SimPopGenerator, self).__init__(use_cache=use_cache, cache_dir=cache_dir,
                                              asynchronous=asynchronous, verbose=verbose)

        self.num_samples = num_samples
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.splitting = splitting
        self.selection = selection
        self.pop_max_size = pop_max_size
        self.num_base_interval = [num_base_min, num_base_max]
        self.feature_max_size = feature_max_size
        self.num_timesteps_interval = [num_timesteps_min, num_timesteps_max]
        self.p_split_interval = [p_split_min, p_split_max]
        self.pop_snps_interval = [pop_snps_min, pop_snps_max]
        self.heritability_interval = [heritability_min, heritability_max]
        self.dom_p_interval = [dom_p_min, dom_p_max]
        self.alpha_interval = [alpha_min, alpha_max]
        self.beta_interval = [beta_min, beta_max]
        self.rr_interval = [rr_min, rr_max]
        self.n_qtl_interval = [n_qtl_min, n_qtl_max]
        self.uo_ratio_interval = [uo_ratio_min, uo_ratio_max]
        self.uo_mask_size = [uo_mask_size_min, uo_mask_size_max]
        self.feature_selection = feature_selection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # First we will rely on the superclass to try to load it from disk if applicable
        r = self.get_item_from_cache(idx)
        if r is not None:
            return r

        start_time = timeit.default_timer()

        pop_max_size = sample_interval(self.pop_max_size) if isinstance(self.pop_max_size, list) else self.pop_max_size

        pop = generate_random_population(do_splitting=self.splitting,
                                         do_selection=self.selection,
                                         num_timesteps=sample_interval(self.num_timesteps_interval),
                                         p_split=sample_interval(self.p_split_interval),
                                         num_base=sample_interval(self.num_base_interval),
                                         pop_size=pop_max_size,
                                         num_snps=sample_interval(self.pop_snps_interval),
                                         heritability=sample_interval(self.heritability_interval),
                                         dom_p=sample_interval(self.dom_p_interval),
                                         alpha=sample_interval(self.alpha_interval),
                                         beta=sample_interval(self.beta_interval),
                                         rr=sample_interval(self.rr_interval),
                                         n_qtl=sample_interval(self.n_qtl_interval),
                                         uo_ratio=sample_interval(self.uo_ratio_interval),
                                         uo_mask_size=sample_interval(self.uo_mask_size),
                                         verbose=self.verbose)

        shuffle(pop)
        pop = pop[:pop_max_size]

        num_train_current = self.num_train_samples if self.eval_sched is None else self.eval_sched[idx]

        train_pop = pop if num_train_current is None else pop[:num_train_current]
        eval_pop = None if num_train_current is None else pop[num_train_current:]

        train_x, train_y = np.stack([p.genotype for p in train_pop]), np.stack([p.phenotypes[0] for p in train_pop])

        if eval_pop is not None:
            eval_x, eval_y = np.stack([p.genotype for p in eval_pop]), np.stack([p.phenotypes[0] for p in eval_pop])

        if self.preprocess:
            if eval_pop is None:
                train_x, train_y = preprocess_population(train_x, train_y, self.feature_max_size,
                                                         self.feature_selection, do_unphase=self.unphase)
            else:
                train_x, train_y, eval_x, eval_y = preprocess_population(train_x, train_y, self.feature_max_size,
                                                                         self.feature_selection, eval_x=eval_x,
                                                                         eval_y=eval_y, do_unphase=self.unphase)

        if eval_pop is None:
            xs = train_x.astype(np.float32)
            ys = train_y.astype(np.float32)
        else:
            xs = np.concatenate((train_x, eval_x), axis=0).astype(np.float32)
            ys = np.concatenate((train_y, eval_y)).astype(np.float32)

        if self.verbose:
            print('Generated a sample in {0:.2f} seconds'.format(timeit.default_timer() - start_time))

        # Use superclass to dump result to cache maybe
        self.write_item_to_cache(idx, xs, ys)

        return xs, ys
