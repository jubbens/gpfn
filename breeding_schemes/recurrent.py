from util.sbwrapper import Trial
from util.breedstrats import phenotypic_selection
import math


class Recurrent:
    def __init__(self, heritability=0.5, n_qtl=50,
                 dom_p=0.1, alpha=0.2, beta=0.5, rr=1.,
                 uo_ratio=None, uo_mask_size=None,
                 selection_intensity=0.01,
                 pregenerations=4):
        self.h2 = [heritability]
        self.n_qtl = n_qtl
        self.pregenerations = pregenerations

        self.dom_p = dom_p
        self.alpha = alpha
        self.beta = beta
        self.rr = rr

        self.uo_ratio = uo_ratio
        self.uo_mask_size = uo_mask_size

        self.selection_intensity = selection_intensity

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

        # Select initial parents
        founders = trial.get_latest_generation()

        num_crosses = int(len(founders) * self.selection_intensity)
        num_children = int(math.ceil(len(founders) / (num_crosses // 2)))

        trial.make_crosses(phenotypic_selection(founders, num_crosses, method='pairs'), num_children=num_children)
        FXs = trial.get_latest_generation()

        for _ in range(self.pregenerations - 1):
            num_crosses = int(len(FXs) * self.selection_intensity)
            num_children = int(math.ceil(len(founders) / (num_crosses // 2)))

            trial.make_crosses(phenotypic_selection(FXs, num_crosses, method='pairs'), num_children=num_children)
            FXs = trial.get_latest_generation()

        training_pop = FXs

        # One more to get the eval pop
        num_crosses = int(len(FXs) * self.selection_intensity)
        num_children = int(math.ceil(len(founders) / (num_crosses // 2)))

        trial.make_crosses(phenotypic_selection(FXs, num_crosses, method='pairs'), num_children=num_children)
        eval_pop = trial.get_latest_generation()

        return training_pop, eval_pop
