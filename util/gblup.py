from util.SeqBreed import selection as sel
import numpy as np


class GBLUP:
    def __init__(self, h2=0.5, ploidy=2):
        self.ploidy = ploidy
        self.h2 = h2

    def get_ebvs(self, x_train, x_test, y_train):
        # If there are multiple traits, use the first
        h2 = self.h2[0] if isinstance(self.h2, list) else self.h2

        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, np.array([-999999.] * x_test.shape[0])))

        G = sel.doGRM(np.transpose(x), nh=self.ploidy)
        ebvs = sel.dogblup(h2, y, G=G, grmFile=None)[x_train.shape[0]:]

        return ebvs
