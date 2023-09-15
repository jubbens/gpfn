from util.sbwrapper import Trial
from util.breedstrats import get_crosses
from random import shuffle


class RIL:
    def __init__(self, num_children, heritability=0.5, n_qtl=50,
                 dom_p=0.1, alpha=0.2, beta=0.5, rr=1.,
                 uo_ratio=None, uo_mask_size=None,
                 descent_generations=3):
        self.num_children = num_children
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

        # Pick two random parents for founders
        founders = trial.get_latest_generation()
        shuffle(founders)
        parents = get_crosses([founders[0], founders[1]], method='pairs')
        trial.make_crosses(parents, num_children=self.num_children)

        FXs = trial.get_latest_generation()

        # Single seed descent
        for _ in range(self.descent_generations):
            trial.make_crosses(get_crosses(FXs, method='selfing'), num_children=1)
            FXs = trial.get_latest_generation()

        return FXs
