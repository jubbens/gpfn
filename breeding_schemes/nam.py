from util.sbwrapper import Trial
from util.breedstrats import get_crosses
from util.tools import get_random_mask, flatten_list
from random import shuffle
import numpy as np


class NAM:
    def __init__(self, num_families=25, family_size=16, train_families=15, heritability=0.5, n_qtl=50,
                 dom_p=0.1, alpha=0.2, beta=0.5, rr=1.,
                 uo_ratio=None, uo_mask_size=None,
                 descent_generations=3):
        self.num_families = num_families
        self.family_size = family_size
        self.train_families = train_families
        self.h2 = [heritability]
        self.n_qtl = n_qtl
        self.descent_generations = descent_generations

        self.dom_p = dom_p
        self.alpha = alpha
        self.beta = beta
        self.rr = rr

        self.uo_ratio = uo_ratio
        self.uo_mask_size = uo_mask_size

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

        # Cross all with one common parent
        founders = trial.get_latest_generation()
        shuffle(founders)
        parents = get_crosses(founders[:self.num_families + 1], method='common')

        # Start with 1 F1 per family
        trial.make_crosses(parents, num_children=1)
        F1s = trial.get_latest_generation()

        # Make family_size F2s
        trial.make_crosses(get_crosses(F1s, method='selfing'), num_children=self.family_size)
        FXs = trial.get_latest_generation()

        # Single seed descent
        for _ in range(self.descent_generations):
            trial.make_crosses(get_crosses(FXs, method='selfing'), num_children=1)
            FXs = trial.get_latest_generation()

        # Sort into families for train/eval split
        assert len(FXs) % self.family_size == 0
        families = [FXs[(i * self.family_size):(i * self.family_size) + self.family_size] for i in range(self.num_families)]
        assert len(families) == self.num_families

        # Split families into train and test
        available_families = list(range(self.num_families))
        mask = get_random_mask(len(available_families), number_unmasked=int(self.train_families))
        train_families = np.array(available_families)[np.array(mask).astype(bool)].tolist()
        eval_families = np.array(available_families)[~np.array(mask).astype(bool)].tolist()

        train = flatten_list([families[i] for i in train_families])
        eval = flatten_list([families[i] for i in eval_families])

        return train, eval
