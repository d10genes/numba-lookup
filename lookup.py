
# coding: utf-8

# In[ ]:

from project_imports import *
# %matplotlib inline


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from pandas.compat import lmap, lfilter, lrange, lzip


# # Imports

# In[ ]:

import numba_lookup as nl; reload(nl); from numba_lookup import *
import tests.test_lookup as ll; reload(ll); from tests.test_lookup import *
import scipy as sp


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


# In[ ]:

k1, k2 = 0, 228
sorted_arr_lookup_ix(ks, vs, ix_table, k1, k2)


# In[ ]:




# In[ ]:

def test_sorted_arr_lookup_ix(dct):
    ks, vs = tup_dct2arr(dct)
    ix_test = get_index(ks)
    test_lookup_eq(partial(sorted_arr_lookup_ix, ks, vs, ix_test), dct)

    
test_sorted_arr_lookup_ix(drand)


# In[ ]:

np.hstack([ks, vs[:, None]])[:15]


# In[ ]:

ks[:15]


# In[ ]:

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

