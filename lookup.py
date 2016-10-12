
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

nm = nmap(drand)


# In[ ]:

try:
    sum_odds_r(nm, rand_keys)
except Exception as e:
    print(e)
    


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

nr.seed(0)
rand_keys_ = nr.randint(len(nm.ks), size=1000)
rand_keys = nm.ks[rand_keys_]


# In[ ]:

@njit
def sum_r(nm, rks):
    s = 0
    for i in range(len(rks)):
        k1, k2 = rks[i]
        s += nm.get(k1, k2)
    return s

@njit
def sum_r2(nm, rks):
    s = 0
    for i in range(len(rks)):
        k1, k2 = rks[i]
        s += nm.get2(k1, k2)
    return s


# In[ ]:




# In[ ]:

del sum_odds_r, sum_odds_r2


# In[ ]:

sum([drand[(k1, k2)] for (k1, k2) in rand_keys])


# In[ ]:

get_ipython().magic('time sum([drand[(k1, k2)] for (k1, k2) in rand_keys])')


# In[ ]:

sum_r2(nm, rand_keys)


# In[ ]:

get_ipython().magic('time sum_r(nm, rand_keys)')
get_ipython().magic('time sum_r2(nm, rand_keys)')


# In[ ]:

get_ipython().magic('timeit sum_r(nm, rand_keys)')
get_ipython().magic('timeit sum_r2(nm, rand_keys)')
# %time sum_odds(nm, rand_keys)

