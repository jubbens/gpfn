from util.sbwrapper import Trial
from util.breedstrats import random_selection, get_crosses


class Silva:
    def __init__(self, num_families=25, family_size=16, heritability=0.5, n_qtl=50,
                 dom_p=0.1, alpha=0.2, beta=0.5, rr=1.,
                 uo_ratio=None, uo_mask_size=None):
        self.num_families = num_families
        self.family_size = family_size
        self.h2 = [heritability]
        self.n_qtl = n_qtl

        self.dom_p = dom_p
        self.alpha = alpha
        self.beta = beta
        self.rr = rr

        self.uo_ratio = uo_ratio
        self.uo_mask_size = uo_mask_size

    def recurrent_block(self, trial):
        # Randomly select initial parents
        founders = trial.get_latest_generation()
        parents = random_selection(founders, self.num_families * 2)

        # Start with 1 F1 per family
        trial.make_crosses(parents, num_children=1)
        F1s = trial.get_latest_generation()

        # Make family_size F2s
        trial.make_crosses(get_crosses(F1s, method='selfing'), num_children=self.family_size)
        F2s = trial.get_latest_generation()

        # Single seed descent
        trial.make_crosses(get_crosses(F2s, method='selfing'), num_children=1)
        F3s = trial.get_latest_generation()

        trial.make_crosses(get_crosses(F3s, method='selfing'), num_children=1)
        F4s = trial.get_latest_generation()

        return F4s

    def forward(self, founders):
        # Import the base population
        trial = Trial()

        trial.set_dom_p(self.dom_p)
        trial.set_gamma_params(self.alpha, self.beta)
        trial.set_rr(self.rr)

        trial.insert_founders(founders, ploidy=2, nchrom=24)
        trial.define_traits(h2=self.h2, nqtl=self.n_qtl)
        trial.make_founder_generation()

        if self.uo_ratio is not None and self.uo_mask_size is not None:
            trial.set_uo_mask(self.uo_ratio, self.uo_mask_size)

        # The F4s from the first recurrent block make up the training population
        training_pop = self.recurrent_block(trial)

        # The testing population is the F4s from the next recurrent block
        eval_pop = self.recurrent_block(trial)

        return training_pop, eval_pop
