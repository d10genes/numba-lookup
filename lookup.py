
# coding: utf-8

# In[ ]:

from project_imports import *
# %matplotlib inline


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# # Imports

# In[ ]:

import scipy as sp
from pandas.compat import lmap, lfilter, lrange, lzip
import numba_lookup as nl; reload(nl); from numba_lookup import *


# In[ ]:

import tests.test_lookup as ll; reload(ll); from tests.test_lookup import *


# ## Sorted array

# N, M = 12000, 1000
# m = sp.sparse.random(N, M, density=.05, format='csc', random_state=1)

# In[ ]:

N, M = 12000, 1000
m = mk_m(n=N, m=M, random_state=1)
drand = coo_todict(m)
ks, vs = nl.tup_dct2arr(drand)


# In[ ]:

ix_table = get_index(ks)


# k1, k2 = 0, 228
# sorted_arr_lookup_ix(ks, vs, ix_table, k1, k2)

# In[ ]:

from numba import jitclass, int64, float64


# In[ ]:

nm = nmap(drand)


# In[ ]:

nmap2dict(nmap(drand)) == drand


# In[ ]:

try:
    sum_odds_r(nm, rand_keys)
except Exception as e:
    print(e)
    


# In[ ]:

nm.get2(0, 86)


# In[ ]:

nm.items()


# In[ ]:

nm2 = mk_nmap2(drand)


# In[ ]:




# In[ ]:

get_index(ks)


# In[ ]:

from numba.numpy_support import from_dtype


# In[ ]:

sum([nm.get(k1, k2) for k1, k2 in keys(ks) if k2 % 2 == 1])


# In[ ]:

lmap(type, list(drand.keys())[0])


# In[ ]:

type(drand[(3149, 598)])


# In[ ]:

list(it.islice(keys(ks), 5))
list(it.islice(values(vs), 5))


# In[ ]:

nr.seed(0)
rand_keys_ = nr.randint(len(nm.ks), size=1000)
rand_keys = nm.ks[rand_keys_]


# In[ ]:

@njit
def sum_odds(nm):
    s = 0
    for k1, k2 in nm.keys():
        if k2 % 2 == 1:
            s += nm.get(k1, k2)
    return s


@njit
def sum_odds2(nm):
    s = 0
    for k1, k2 in nm.keys():
        if k2 % 2 == 1:
            s += nm.get2(k1, k2)
    return s


# In[ ]:

get_ipython().magic('time sum([v for (k1, k2), v in drand.items() if k2 % 2 == 1])')


# In[ ]:

@njit
def sum_r(nm, rks):
    s = 0
    for i in range(len(rks)):
        k1, k2 = nm.ks[i]
        s += nm.get(k1, k2)
    return s

@njit
def sum_r2(nm, rks):
    s = 0
    for i in range(len(rks)):
        k1, k2 = nm.ks[i]
        s += nm.get2(k1, k2)
    return s


# In[ ]:

del sum_odds_r, sum_odds_r2


# In[ ]:

get_ipython().magic('timeit sum_r(nm, rand_keys)')
get_ipython().magic('timeit sum_r2(nm, rand_keys)')
# %time sum_odds(nm, rand_keys)


# In[ ]:

@jitclass([
        ('ks', int64[:, :]),      
#         ('k1s', int64[:]),      
#         ('k2s', int64[:]),      
        ('vs', float64[:]),      
        ('ix_table', int64[:, :]),      
])
class NMap(object):
#     def __init__(self, k1s, k2s, vs, ix_table):  # ix_table
    def __init__(self, ks, vs, ix_table):  # ix_table
#         self.k1s = k1s
#         self.k2s = k2s
        self.ks = ks
        self.vs = vs
        self.ix_table = ix_table
#         self.ix_table = get_index(ks)
        
    def get(self, k1, k2):
        # k1, k2 = tup
        return sorted_arr_lookup_ix(self.ks, self.vs, self.ix_table, k1, k2)
    
    def __getitem__(self, tup):
        k1, k2 = tup
        return sorted_arr_lookup_ix(self.ks, self.vs, self.ix_table, k1, k2)
#     def __getitem__(self, tup):
#         k1, k2 = tup
#         return sorted_arr_lookup_ix(self.ks, self.vs, self.ix_table, k1, k2)

k1s_ = ks[:, 0]
k2s_ = ks[:, 1]
try:
#     NMap(ks, vs, get_index(ks))
    nm = NMap(ks, vs, get_index(ks))
#     nm = NMap(ks, vs)
except Exception as e:
    print(e)


# In[ ]:

nm.get(0, 31)


# In[ ]:

nm[0, 31]


# In[ ]:

nm.ks


# In[ ]:

sorted_arr_lookup_ix


# In[ ]:

@njit
def sorted_arr_lookup_ix(k1s, k2s, vals, ix_table, k1, k2):
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
        ix_k2 = len(k1s)
    else:
        ix_k2 = lookup_ix(ix_table, k1 + 1, check=False)

    c2 = k2s[ix_k1:ix_k2]
    ixb1 = np.searchsorted(c2, k2)
    # ixb2 = np.searchsorted(c2, k2 + 1)

    ix = ix_k1 + ixb1
    k1_, k2_ = k1s[ix], k2s[ix]

    if (k1_ != k1) or (k2_ != k2):
        print('k1', k1, 'k2', k2)
        print(k1_, k2_)
        raise KeyError("Array doesn't contain keys")
    return vals[ix]


# In[ ]:




# In[ ]:

# %time arr = nl.tup_dct2arr(drand)
get_ipython().magic('time ks, vs = nl.tup_dct2arr(drand)')


# In[ ]:

vs[unq_ix]


# In[ ]:

ix


# In[ ]:

ks[:, 0]


# In[ ]:

ks


# In[ ]:

ix_table = get_index


# In[ ]:




# In[ ]:

vs


# In[ ]:

ks


# In[ ]:

arr_ = np.array([[a, b, c] for (a, b), c in drand.items()])
arr = np.sort(arr_, axis=0)


# In[ ]:

for dim in range(3):
    assert set(arr[:, dim]) == set(arr_uns[:, dim])


# In[ ]:

drand[(0, 18)]


# In[ ]:

DataFrame(arr)


# In[ ]:

DataFrame


# In[ ]:

arr_[:, :]


# In[ ]:

a


# In[ ]:

np.lexsort(a.T, axis=-1)


# In[ ]:

DataFrame(a)


# In[ ]:

DataFrame(a[:, :-1].T)


# In[ ]:

a[:, :-1].T


# In[ ]:

l = [2,3,1,4]
l[-1:0:-1]


# In[ ]:




# In[ ]:

sa = a[np.lexsort(a[:, :-1].T, axis=0)]
DataFrame(sa)


# In[ ]:

a = arr_[:10]
DataFrame(a, )


# In[ ]:

drand[(259, 70)]


# In[ ]:

DataFrame(np.sort(a, axis=0))


# In[ ]:

len(drand)


# In[ ]:

arr.shape


# In[ ]:

arr_uns


# In[ ]:

get_ipython().magic('time ss = sorted(its)')


# ss = sorted(drand.items())

# In[ ]:

ks, vs = zip(*ss)


# In[ ]:

np.array(ks)


# In[ ]:

np.array(vs)


# In[ ]:

m


# In[ ]:

m[1]


# In[ ]:

test_sorted_arr_lookup_ix(drand)


# In[ ]:

test_sorted_arr_lookup(drand)


# In[ ]:

arr


# In[ ]:

np.array([[r, c, m[r, c]] for r, c in lzip(*m.nonzero())])


# In[ ]:

m


# In[ ]:

{(0, 1): .957516}


# In[ ]:

DataFrame(m.toarray())

