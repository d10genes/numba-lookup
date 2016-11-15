
# coding: utf-8

# - limitations
#     - can't construct with numba

# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[ ]:

from project_imports import *


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "var csc = IPython.keyboard_manager.command_shortcuts\ncsc.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\ncsc.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\ncsc.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# # Imports

# In[ ]:

import scipy as sp
from pandas.compat import lmap, lfilter, lrange, lzip
import numba_lookup as nl; reload(nl); from numba_lookup import *


# In[ ]:

import tests.test_lookup as ll; reload(ll); from tests.test_lookup import *


# # Benchmarks: constant vs linear vs log

# In[ ]:

import timeit

def run_inputs(f=None, genargs=None, inputs=[], n_repeat=10):
    ts = OrderedDict()
    for n in inputs:
        test_func = lambda: f(*genargs(n))
        ts[n] = timeit.timeit(test_func, number=n_repeat) / n_repeat
    s = Series(ts, name='Time')
    s.index.name = 'N'
    return s

def s2df(s, **kw):
    "Series to DataFrame"
    df = s.reset_index(drop=0)
    for k, v in kw.items():
        df[k] = v
    return df


# ## Constant time query

# In[ ]:

s = Series([1,2], name='a')
s = s.reset_index(drop=0)
s


# In[ ]:

def mk_rnd_dct_arr(dictsize, nkeys=1000):
    "Generate random int dict and random subset of keys"
    rx = nr.randint(0, int(1e9), size=(dictsize, 2))
    return dict(rx), nr.choice(rx[:, 0], size=nkeys)

def dct_benchmark(dct, ks):
    "Sum all of the values in dict for given keys"
    return sum(dct[k] for k in ks)


# In[ ]:

ts_const = run_inputs(f=dct_benchmark, genargs=mk_rnd_dct_arr, inputs=[1, 10, 100, 1000, 10000, ]) 
ts_const


# ## Linear lookup

# In[ ]:

def dct2linear_lookup(dct):
    return np.array(list(dct.items()))

def mk_rnd_linear_dct(n, nkeys=1000):
    d, a = mk_rnd_dct_arr(n, nkeys=nkeys)
    return dct2linear_lookup(d), a

def linear_lookup_get(arr, kquery):
    for k, v in arr:
        if k == kquery:
            return v
    raise KeyError(kquery)
    
def linear_benchmark(dct, ks):
    return sum(linear_lookup_get(dct, k) for k in ks)


# In[ ]:

d, a = mk_rnd_dct_arr(100, 1000)
dlin = dct2linear_lookup(d)

assert dct_benchmark(d, a) == linear_benchmark(dlin, a)


# In[ ]:

ts_lin = run_inputs(f=linear_benchmark, genargs=mk_rnd_linear_dct, inputs=[1, 10, 100], n_repeat=10) 
ts_lin


# ### Numba speedup

# In[ ]:

@njit
def linear_lookup_get_nb(arr, kquery):
    for i in range(len(arr)):
        if arr[i, 0] == kquery:
            return arr[i, 1]
    print(kquery)
    raise KeyError

@njit
def linear_benchmark_nb(dct, ks):
    s = 0
    for k in ks:
        s += linear_lookup_get_nb(dct, k)
    return s


# In[ ]:

assert linear_benchmark(dlin, a) == linear_benchmark_nb(dlin, a)


# In[ ]:

ts_lin_nb = run_inputs(f=linear_benchmark_nb, genargs=mk_rnd_linear_dct, inputs=10 ** np.arange(6), n_repeat=10) 
ts_lin_nb


# In[ ]:




# In[ ]:

import seaborn as sns


# In[ ]:

times


# In[ ]:

times = pd.concat([
    s2df(ts_const, Complexity='Constant'),
    s2df(ts_lin, Complexity='Linear'),
    s2df(ts_lin_nb, Complexity='Linear numba'),
])

def plot(x, y, **_):
    return plt.plot(x, y, '.:')

g = sns.FacetGrid(times, col='Complexity', sharex=False)
g.map(plot, 'N', 'Time')


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

