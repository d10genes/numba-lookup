
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


# ### Constant
# 
# import time
# timef = time.perf_counter
# 
# def timer(f):
#     def f2(*a, **k):
#         st = timef()
#         res = f(*a, **k)
#         t = timef() - st
#         # print(t)
#         return t, res
#     return f2
# 
# 

# In[ ]:

import timeit

def run_inputs(f=None, genargs=None, inputs=[], n_repeat=10):
    ts = OrderedDict()
    for n in inputs:
        args = genargs(n)
        ts[n] = timeit.timeit(lambda: f(*args), number=n_repeat) / n_repeat
    return Series(ts)


# ### Constant time query

# In[ ]:

def mk_rnd_dct_arr(n, asize=1000):
    rx = nr.randint(0, int(1e9), size=(n, 2))
    return dict(rx), nr.choice(rx[:, 0], size=asize)

def dct_f(dct, ks):
    return sum(dct[k] for k in ks)


# In[ ]:

ts = run_inputs(f=dct_f, genargs=mk_rnd_dct_arr, inputs=[1, 10, 100, 1000, 10000, ]) 
ts


# ### Linear lookup

# In[ ]:

def dct2linear_lookup(dct):
    return np.array(list(dct.items()))

def mk_rnd_linear_dct(n, asize=1000):
    d, a = mk_rnd_dct_arr(n, asize=asize)
    return dct2linear_lookup(d), a

def linear_lookup_get(arr, kquery):
    for k, v in arr:
        if k == kquery:
            return v
    raise KeyError(kquery)
    
def linear_f(dct, ks):
    return sum(linear_lookup_get(dct, k) for k in ks)


# In[ ]:

d, a = mk_rnd_dct_arr(100, 1000)
dlin = dct2linear_lookup(d)

assert dct_f(d, a) == linear_f(dlin, a)


# In[ ]:

ts2 = run_inputs(f=linear_f, genargs=mk_rnd_linear_dct, inputs=[1, 10, 100], n_repeat=10) 
ts2


# In[ ]:

@njit
def linear_lookup_get_nb(arr, kquery):
    for i in range(len(arr)):
        if arr[i, 0] == kquery:
            return arr[i, 1]
    print(kquery)
    raise KeyError

@njit
def linear_f_nb(dct, ks):
    s = 0
    for k in ks:
        s += linear_lookup_get_nb(dct, k)
    return s


# In[ ]:

assert linear_f(dlin, a) == linear_f_nb(dlin, a)


# In[ ]:

ts3 = run_inputs(f=linear_f_nb, genargs=mk_rnd_linear_dct, inputs=10 ** np.arange(6), n_repeat=10) 
Series(ts3)


# In[ ]:




# In[ ]:

@njit
def linear_lookup_get_nb(arr, kquery):
    for k, v in arr:
        if k == kquery:
            return v
    raise KeyError(kquery)

    
@njit
def linear_f_nb(dct, ks):
    s = 0
    for k in ks:
        s += linear_lookup_get_nb(dct, k)
    return s


# In[ ]:




# In[ ]:



try:
#     linear_lookup_get_nb(d, 1.)
    linear_f_nb(d, a)
except Exception as e:
    print(e)


# In[ ]:

assert linear_f(dlin, a) == linear_f_nb(dlin, a)


# In[ ]:

assert dct_f(d, a) == linear_f_nb(dlin, a)
assert linear_f(dlin, a) == linear_f_nb(dlin, a)


# In[ ]:




# In[ ]:

ts3 = run_inputs(f=linear_f_nb, genargs=mk_rnd_linear_dct, inputs=[1, 10, 100, 1000, 10000], n=1) 
Series(ts3)


# In[ ]:


Series(ts2)


# In[ ]:

ts2 = run_inputs(f=linear_f, genargs=mk_rnd_linear_dct, inputs=[1, 10, 100], n=2) 
Series(ts2)


# In[ ]:

Series(ts).reset_index(drop=0).rename(columns={'index': 'Input size', 0: 'Avg time'})


# In[ ]:

dct_get(d, a)


# In[ ]:




# In[ ]:

t = timeit.Timer('dct_get(d, a)')
t.timeit(number=100)


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

