'''
Logarithmic time lookup with double keys using numba
'''
from numba import njit, jitclass, int64, float64
from numba.numpy_support import from_dtype
import numpy as np


nx = lambda x: next(iter(x))


def tup_dct2arr(dct, kv=True):
    arr = np.array([[a, b, c] for (a, b), c in dct.items()])
    _, ncols = arr.shape
    arrs = arr[np.lexsort([arr[:, col] for col in
                           range(ncols - 1)][::-1])]

    if not kv:
        return arrs
    ks = arrs[:, :-1].astype(int)
    vs = arrs[:, -1]
    return ks, vs


@njit
def sorted_arr_lookup(ks, vs, k1, k2):
    """A is a n x 3 array with the first 2 columns sorted.
    The values are in the 3rd column.
    The lookup uses a binary sort on the first 2 columns to
    get the value in the third.
    """
    c1 = ks[:, 0]
    ixa1 = np.searchsorted(c1, k1)
    ixa2 = np.searchsorted(c1, k1 + 1)

    c2 = ks[ixa1:ixa2, 1]
    ixb1 = np.searchsorted(c2, k2)

    ix = ixa1 + ixb1
    k1_, k2_ = ks[ix]

    if (k1_ != k1) or (k2_ != k2):
        print(k1, k2, k1_, k2_)
        raise KeyError("Array doesn't contain keys")
    v = vs[ix]
    return v


# Use precomputed index to find 1st key faster
def get_index(ks):
    """Return lookup table for indices of each
    new entry in 1st column
    [0, blah, ...]
    [1, blah, ...]
    [1, blah, ...]
    [2, blah, ...]
    ->
    val first_ix
    [0, 0]
    [1, 1]
    [2, 3]
    """
    unq_vals, unq_ix = np.unique(ks[:, 0], return_index=True)
    return np.vstack([unq_vals, unq_ix]).T


@njit
def lookup_ix(arr, ix1, check=False):
    ix = np.searchsorted(arr[:, 0], ix1)
    v_, v_ix = arr[ix]
    if check and (v_ != ix1):
        print(ix1, v_)
        raise KeyError
    return v_ix


@njit
def sorted_arr_lookup_ix(karr, vals, ix_table, k1, k2):
    """A is a n x 3 array with the first 2 columns sorted.
    The values are in the 3rd column.
    The lookup uses a binary sort on the first 2 columns to
    get the value in the third.
    ix_table: nx2 array.
        - 1st col is deduplicated, sorted k1 values
        - 2nd col is index in `karr` of first occurrence of row's k1
    """
    # print(k1, k2)
    mx_index = ix_table[-1, 0]
    ix_k1 = lookup_ix(ix_table, k1)
    if k1 == mx_index:
        ix_k2 = len(karr)
    else:
        ix_k2 = lookup_ix(ix_table, k1 + 1, check=False)

    c2 = karr[ix_k1:ix_k2, 1]
    ixb1 = np.searchsorted(c2, k2)
    # ixb2 = np.searchsorted(c2, k2 + 1)

    ix = ix_k1 + ixb1
    k1_, k2_ = karr[ix]

    if (k1_ != k1) or (k2_ != k2):
        print('k1', k1, 'k2', k2)
        print(k1_, k2_)
        raise KeyError("Array doesn't contain keys")
    return vals[ix]


################
# Dictwrappers #
################
@jitclass([
    ('ks', int64[:, :]),
    ('vs', float64[:]),
    ('ix_table', int64[:, :]),
])
class NMap(object):

    def __init__(self, ks, vs, ix_table):  # ix_table
        self.ks = ks
        self.vs = vs
        self.ix_table = ix_table

    def get(self, k1, k2):
        return sorted_arr_lookup_ix(self.ks, self.vs,
                                    self.ix_table, k1, k2)


@jitclass([
    ('ks', int64[:, :]),
    ('vs', float64[:]),
    ('ix_table', int64[:, :]),
])
class NMap2(object):
    "Testing a different strategy..."
    def __init__(self, ks, vs):  # ix_table
        self.ks = ks
        self.vs = vs

    def get(self, k1, k2):
        return sorted_arr_lookup(self.ks, self.vs, k1, k2)


def nmap(dct=None, ks=None, vs=None):
    "Constructor for lookup wrapper."
    if ks is None or vs is None:
        ks, vs = tup_dct2arr(dct)
    ix = get_index(ks)

    spec = [
        ('ks', from_dtype(ks.dtype)[:, :]),
        ('vs', from_dtype(vs.dtype)[:]),
        ('ix_table', from_dtype(ix.dtype)[:, :]),
    ]

    @jitclass(spec)
    class NMap(object):

        def __init__(self, ks, vs, ix_table):  # ix_table
            self.ks = ks
            self.vs = vs
            self.ix_table = ix_table

        # def get(self, k1, k2=None):
        def get(self, ks):
            k1, k2 = ks
            return sorted_arr_lookup_ix(self.ks, self.vs, self.ix_table, k1, k2)

        # def get2(self, k1, k2):
        def get2(self, ks):
            k1, k2 = ks
            return sorted_arr_lookup(self.ks, self.vs, k1, k2)

        def keys(self):
            r = []
            for i in range(len(ks)):
                r.append((ks[i, 0], ks[i, 1]))
            return r

        def values(self):
            r = []
            for i in range(len(vs)):
                r.append(vs[i])
            return r

        def items(self):
            r = []
            for k, v in zip(self.keys(), self.values()):
                r.append((k, v))
            return r

    return NMap(ks, vs, ix)


def mk_nmap2(dct=None, ks=None, vs=None):
    if ks is None or vs is None:
        ks, vs = tup_dct2arr(dct)
    return NMap2(ks, vs)


def nmap2dict(nm):
    return dict(nm.items())
