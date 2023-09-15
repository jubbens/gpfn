from util.SeqBreed import genome as gg
import numpy as np
import warnings


def _phase(x):
    # At locations where x is zero, m and f are both zero
    m = np.zeros_like(x)
    f = np.zeros_like(x)

    # At locations where x is two, m and f are both one
    m[x == 2] = 1
    f[x == 2] = 1

    # At locations where x is one, we need to pick whether the major allele is on m or f
    to_choose = np.argwhere(x == 1)
    np.random.shuffle(to_choose)
    half = int(to_choose.shape[0] / 2)
    m_half, f_half = to_choose[:half], to_choose[half:]
    m[[p[0] for p in m_half], [p[1] for p in m_half]] = 1
    f[[p[0] for p in f_half], [p[1] for p in f_half]] = 1

    return np.concatenate((m, f), axis=0).T


class GFounderMock:
    """A mock object for GFounder in the case of a random population"""
    def __init__(self, nsnp, nbase, weights=None, ploidy=2):
        if weights is None:
            # Random H and MAF
            het = np.random.uniform(0.05, 0.15)
            minor = np.random.uniform(0.1, 0.30)
            major = 1. - (het + minor)
            weights = [minor, het, major]

        g = np.random.choice(3, (nbase, nsnp), p=weights)
        g = _phase(g)

        self.nbase = nbase
        self.ploidy = ploidy
        self.nsnp = nsnp
        self.g = g

        self.f = self.g.mean(axis=1)
        self.f[np.where(self.f > 0.5)] = 1. - self.f[np.where(self.f > 0.5)]


class GenomeMock:
    """A mock object for Genome in the case of a random population"""
    def __init__(self, num_snp, nchr, ploidy=2, autopolyploid=True, rr=1.):
        self.nchr = nchr
        self.num_snp = num_snp
        self.ploidy = ploidy
        self.autopolyploid = autopolyploid

        chr_names = [str(i) for i in range(nchr)]
        self.dictionar = dict(zip(chr_names, range(self.nchr)))

        snp_per_chrom = int(num_snp / nchr)
        bounds = np.arange(0, num_snp, snp_per_chrom)
        sd = [snp_per_chrom] * (nchr - 1)
        sd.append(snp_per_chrom + (num_snp % snp_per_chrom))

        self.chrs = [gg.Chromosome(name=chr_names[i], pos=bounds[i], nsnp=sd[i], length=sd[i],
                                   xchr=False, ychr=False, mtchr=False, cM2Mb=rr)
                     for i in range(nchr)]

        nsnp = list(self.chrs[i].nsnp for i in range(self.nchr))
        self.cumpos = np.cumsum(nsnp) - nsnp


class Individual:
    def __init__(self):
        self.id = None
        self.phenotypes = []
        self.genotype = None
        self.dam = None
        self.sire = None


class Trial:
    def __init__(self):
        self._generation_bounds = []
        self._pop = None
        self._gbase = None
        self._gfeatures = None
        self._traits = None
        self._pedfile = None
        self._seqfile = None
        self._ploidy = None
        self._chipseq = None

        self.dom_p = 0.
        self.alpha = 0.2
        self.beta = 5.
        self.rr = 1.

        self.X = {}

        self.uo_mask = None

    def set_dom_p(self, p):
        self.dom_p = p

    def set_gamma_params(self, a, b):
        self.alpha = a
        self.beta = b

    def set_rr(self, rate):
        self.rr = rate

    def set_uo_mask(self, uo_ratio, uo_mask_size):
        mask = np.ones(self.get_num_snps())
        grid = np.arange(mask.shape[0])
        starts = np.random.choice(grid, size=int((grid.shape[0] * uo_ratio) / uo_mask_size), replace=False)
        ends = starts + uo_mask_size
        ends[ends + 1 > grid.shape[0]] = grid.shape[0] - 1

        for start, end in zip(starts, ends):
            mask[start:end + 1] = 0

        self.uo_mask = mask

    def get_num_snps(self):
        if self.uo_mask is None:
            return self._gbase.nsnp
        else:
            return int(np.sum(self.uo_mask))

    def get_generation(self, gen_num):
        bounds = self._generation_bounds[gen_num]
        gen = []

        for i in range(bounds[0], bounds[1]):
            ind = Individual()
            ind.id = self._pop.inds[i].id
            ind.phenotypes = self._pop.inds[i].y
            ind.genotype = self.X[ind.id]

            ind.dam = self._pop.inds[i].id_dam
            ind.sire = self._pop.inds[i].id_sire

            # Redact if we have a mask
            if self.uo_mask is not None:
                if ind.genotype.shape != self.uo_mask.shape:
                    # Hack for something weird that happens inside of SB
                    # For some reason the size of the genotype data is sometimes a few less than nsnp?
                    self.uo_mask = self.uo_mask[:ind.genotype.shape[0]]

                ind.genotype = np.extract(self.uo_mask, ind.genotype)

            gen.append(ind)

        return gen

    def prune_inds(self, ids):
        self._pop.inds = [self._pop.inds[iid - 1] for iid in ids]
        self._pop.n = len(ids)
        self._generation_bounds = [(0, len(ids))]

        # Reset all of the individual IDs
        for i in range(1, len(ids) + 1):
            self._pop.inds[i - 1].id = i

    def get_latest_generation(self):
        return self.get_generation(-1)

    def get_all_generations(self):
        return [self.get_generation(i) for i in range(len(self._generation_bounds))]

    def get_all_individuals(self):
        return [i for gen in self.get_all_generations() for i in gen]

    def get_number_of_generations(self):
        return len(self._generation_bounds)

    def insert_founders(self, founders, ploidy, nchrom):
        # Make sure the founders data is in a form we expect
        assert np.max(founders) == 2. and np.min(founders) == 0., 'inserted data should be normalized to [0, 2]'

        nbase, nsnp = founders.shape

        self._gbase = GFounderMock(nsnp, nbase, ploidy=ploidy)

        self._gbase.g = _phase(founders)
        self._gbase.f = self._gbase.g.mean(axis=1)
        self._gbase.f[np.where(self._gbase.f > 0.5)] = 1. - self._gbase.f[np.where(self._gbase.f > 0.5)]

        self._gfeatures = GenomeMock(nsnp, nchrom, ploidy=ploidy, rr=self.rr)
        self._pedfile = None
        self._seqfile = None
        self._ploidy = ploidy

        if self._seqfile is None:
            self._chipseq = gg.Chip(gfounders=self._gbase, genome=self._gfeatures, nsnp=self._gfeatures.num_snp, name='seq_chip')
        else:
            self._chipseq = gg.Chip(chipFile=self._seqfile, genome=self._gfeatures, name='seq_chip')

    def import_founder_data(self, genfile, ploidy, pedfile=None, seqfile=None):
        self._gbase = gg.GFounder(vcfFile=genfile, snpFile=seqfile, ploidy=ploidy)
        self._gfeatures = gg.Genome(snpFile=seqfile, ploidy=ploidy)
        self._pedfile = pedfile
        self._seqfile = seqfile
        self._ploidy = ploidy

        if self._seqfile is None:
            self._chipseq = gg.Chip(gfounders=self._gbase, genome=self._gfeatures, nsnp=self._gbase.nsnp, name='seq_chip')
        else:
            self._chipseq = gg.Chip(chipFile=self._seqfile, genome=self._gfeatures, name='seq_chip')

    def generate_random_founders(self, nsnp, nbase, ploidy=2, nchrom=24):
        self._gbase = GFounderMock(nsnp, nbase, ploidy=ploidy)
        self._gfeatures = GenomeMock(nsnp, nchrom, ploidy=ploidy, rr=self.rr)
        self._pedfile = None
        self._seqfile = None
        self._ploidy = ploidy

        if self._seqfile is None:
            self._chipseq = gg.Chip(gfounders=self._gbase, genome=self._gfeatures, nsnp=self._gfeatures.num_snp, name='seq_chip')
        else:
            self._chipseq = gg.Chip(chipFile=self._seqfile, genome=self._gfeatures, name='seq_chip')

    def define_traits(self, h2, nqtl=None, qtl_file=None):
        if nqtl is not None:
            self._traits = gg.QTNs(h2=h2, genome=self._gfeatures, nqtn=nqtl,
                                   dom_p=self.dom_p, alpha=self.alpha, beta=self.beta)
        elif qtl_file is not None:
            self._traits = gg.QTNs(h2=h2, genome=self._gfeatures, qtnFile=qtl_file,
                                   dom_p=self.dom_p, alpha=self.alpha, beta=self.beta)
        else:
            warnings.warn('Need either nqtl or qtl_file specified.')
            exit()

    def make_founder_generation(self):
        self._pop = gg.Population(self._gfeatures, pedFile=self._pedfile, generation=None, qtns=self._traits, gfounders=self._gbase)

        self._generation_bounds.append((0, self._pop.n))
        self.__update_X(0, self._pop.n)

    def __update_X(self, low, high):
        added = gg.do_X(self._pop.inds[low:high], self._gfeatures, self._gbase, chip=self._chipseq)

        for id, idx in zip([ind.id for ind in self._pop.inds[low:high]], range(added.shape[0])):
            self.X[id] = added[:, idx]

    def trim_X(self, keep):
        self.X = {k: self.X[k] for k in keep}

    def make_crosses(self, crosses, num_children):
        if not crosses:
            raise Exception('Trying to make an empty set of crosses?')

        low = self._generation_bounds[-1][1]

        if isinstance(num_children, int):
            # We are doing a constant number of children
            high = low + (len(crosses) * num_children)
            self._generation_bounds.append((low, high))

            num_children = range(num_children)

            for cross in crosses:
                for _ in num_children:
                    parents = [self._pop.inds[cross[0] - 1], self._pop.inds[cross[1] - 1]]  # -1 because this is zero-indexed
                    assert parents[0].id == cross[0]
                    assert parents[1].id == cross[1]

                    self._pop.addInd(parents, genome=self._gfeatures, gfounders=self._gbase, qtns=self._traits, id=None, sex=None, t=None)
        else:
            # We are doing a variable number of children per cross
            high = low + np.sum(num_children)
            self._generation_bounds.append((low, high))

            for cross, nc in zip(crosses, num_children):
                for i in range(nc):
                    parents = [self._pop.inds[cross[0] - 1], self._pop.inds[cross[1] - 1]]  # -1 because this is zero-indexed
                    self._pop.addInd(parents, genome=self._gfeatures, gfounders=self._gbase, qtns=self._traits, id=None, sex=None, t=None)

        self.__update_X(low, high)
