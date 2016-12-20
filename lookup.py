
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
import seaborn as sns
from pandas.compat import lmap, lfilter, lrange, lzip
import numba_lookup as nl; reload(nl); 
# from numba_lookup import *
(";")


# In[ ]:

import tests.test_lookup as ll; reload(ll); from tests.test_lookup import *


# # Benchmarks: constant vs linear vs log

# In[ ]:

import timeit

def run_benchmark(f=None, genargs=None, inputs=[], n_repeat=10):
    ts = OrderedDict()
    for n in inputs:

        def setup_arg():
            """Gotta do weird stuff for `timeit` to see things
            correctly in its namespace."""
            setup_arg.arg = genargs(n)
            
        test_func = lambda: f(*setup_arg.arg)
        ts[n] = timeit.timeit(test_func, number=n_repeat, setup=setup_arg) / n_repeat
        
#         test_func = lambda: f(*genargs(n))
#         ts[n] = timeit.timeit(test_func, number=n_repeat) / n_repeat
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

# def mk_rnd_dct_arr1d(dictsize, nkeys=1000):
#     "Generate random int dict and random subset of keys"
#     rx = nr.randint(0, int(1e9), size=(dictsize, 2))
#     return dict(rx), nr.choice(rx[:, 0], size=nkeys)
# 
# def dct_benchmark1d(dct, ks):
#     "Sum all of the values in dict for given keys"
#     return sum(dct[k] for k in ks)

# In[ ]:

def mk_rnd_dct_arr2d(dictsize, nkeys=1000):
    "Generate random int dict and random subset of keys"
    rx = nr.randint(0, int(1e9), size=(dictsize, 3))
    d = {(k1, k2): v for k1, k2, v in rx}
    ks = [(rx[i, 0], rx[i, 1]) for i in nr.choice(np.arange(dictsize), size=nkeys)]
    return d, ks

def dct_benchmark(dct, ks):
    "Sum all of the values in dict for given keys"
    return sum(dct.get(k) for k in ks)


# In[ ]:

ts_const = run_benchmark(f=dct_benchmark, genargs=mk_rnd_dct_arr2d, inputs=[1, 10, 100, 1000, 10000, 100000]) 
ts_const


# ## Linear lookup

# def dct2linear_lookup(dct):
#     return np.array(list(dct.items()))
# 
# def mk_rnd_linear_dct(n, nkeys=1000):
#     d, a = mk_rnd_dct_arr(n, nkeys=nkeys)
#     return dct2linear_lookup(d), a
# 
# def linear_lookup_get(arr, kquery):
#     for k, v in arr:
#         if k == kquery:
#             return v
#     raise KeyError(kquery)
#     
# def linear_benchmark(dct, ks):
#     return sum(linear_lookup_get(dct, k) for k in ks)

# In[ ]:

DataFrame([[3, 4, 1], [2, 3, 4]], columns=['Key 1', 'Key 2', 'Value'])


# In[ ]:

class DWrapper:
    def __init__(self, dct, getter=None):
        self.dct = dct
        self.getter = getter
        
    def get(self, k):
        return self.getter(self.dct, k)


# In[ ]:

def dct2linear_lookup2d(dct: dict, getter=None):
    "Convert dict to linear array lookup"
    lookup = np.array([(k1, k2, v) for (k1, k2), v in dct.items()])
    return DWrapper(lookup, getter=getter)

def mk_rnd_linear_dct(n, nkeys=1000, getter=None):
    d, a = mk_rnd_dct_arr2d(n, nkeys=nkeys)
    return dct2linear_lookup2d(d, getter=getter), a

def linear_lookup_get2d(arr, kquery):
    for k1, k2, v in arr:
        if (k1, k2) == kquery:
            return v
    raise KeyError(kquery)
    
# def linear_benchmark(dct, ks):
#     return sum(linear_lookup_get2d(dct, k) for k in ks)


# In[ ]:

# Check equiv
d, a = mk_rnd_dct_arr2d(100, 1000)
dlin = dct2linear_lookup2d(d, getter=linear_lookup_get2d)

# assert dct_benchmark(d, a) == linear_benchmark(dlin, a)
assert dct_benchmark(d, a) == dct_benchmark(dlin, a)


# In[ ]:

ts_lin = run_benchmark(f=dct_benchmark, genargs=partial(mk_rnd_linear_dct, getter=linear_lookup_get2d),
                       inputs=[1, 10, 100], n_repeat=10) 
ts_lin


# ### Numba speedup

# @njit
# def linear_lookup_get_nb(arr, kquery):
#     for i in range(len(arr)):
#         if arr[i, 0] == kquery:
#             return arr[i, 1]
#     print(kquery)
#     raise KeyError
# 
# @njit
# def linear_benchmark_nb(dct, ks):
#     s = 0
#     for k in ks:
#         s += linear_lookup_get_nb(dct, k)
#     return s

# In[ ]:

from numba import njit


# In[ ]:

@njit
def linear_lookup_get_nb2d(arr, kquery):
    for i in range(len(arr)):
        if (arr[i, 0], arr[i, 1]) == kquery:
            return arr[i, 2]
    print(kquery)
    raise KeyError


# In[ ]:

# Check equiv
dlin_nb = dct2linear_lookup2d(d, getter=linear_lookup_get_nb2d)

assert dct_benchmark(dlin, a) == dct_benchmark(dlin_nb, a)
# assert linear_benchmark(dlin, a) == linear_benchmark_nb(dlin, a)


# In[ ]:

_gen_nb = partial(mk_rnd_linear_dct, getter=linear_lookup_get_nb2d)
ts_lin_nb = run_benchmark(f=dct_benchmark, genargs=_gen_nb,
                          inputs=10 ** np.arange(6), n_repeat=10)
ts_lin_nb


# ts_lin_nb = run_benchmark(f=dct_benchmark, genargs=mk_rnd_linear_dct, inputs=[1, 10, 100], n_repeat=10) 
# ts_lin_nb

# ts_lin_nb = run_benchmark(f=linear_benchmark_nb, genargs=mk_rnd_linear_dct, inputs=10 ** np.arange(6), n_repeat=10) 
# ts_lin_nb

# In[ ]:

times = pd.concat([
    s2df(ts_const, Complexity='Constant'),
    s2df(ts_lin, Complexity='Linear'),
    s2df(ts_lin_nb, Complexity='Linear numba'),
])


# In[ ]:

times_scales = pd.concat([
    s2df(times.set_index('N'), Scale='Linear'),
    s2df(times.set_index('N'), Scale='Log'),
])


# In[ ]:

def plot(x, y, scales, **kw):
    scale, *_ = scales
#     print(scale)
#     print(kw)
    p = plt.plot(x, y, '.:')
    if scale == 'Log':
        plt.xscale('log')
    return p

g = sns.FacetGrid(times_scales, col='Complexity', row='Scale', sharex=False)
g.map(plot, 'N', 'Time', 'Scale');


# def plot(x, y, **kw):
#     p = plt.plot(x, y, '.:')
#     plt.xscale('log')
#     return p
# 
# g = sns.FacetGrid(times, col='Complexity', sharex=False)
# g.map(plot, 'N', 'Time');

# def plot_zoom(x, y, **_):
# #     global x1, y1
# #     x1, y1 = x, y
#     bm = x <= 1000
#     p = plt.plot(x[bm], y[bm], '.:')
# #     plt.xlim(None, 200)
#     return p
# 
# g = sns.FacetGrid(times, col='Complexity', sharex=False)
# g.map(plot_zoom, 'N', 'Time');

# ## Benchmark Log lookup

# import scipy as sp
# from pandas.compat import lmap, lfilter, lrange, lzip
# import numba_lookup as nl; reload(nl); from numba_lookup import *
# ;;

# @njit
# def benchmark_nb(dct, ks):
#     s = 0
#     for k in ks:
#         s += dct.get(k)
#     return s

# In[ ]:

from numba_lookup import nmap


# In[ ]:

dlog = nmap(d)

# Check equivalence
assert dct_benchmark(d, a) == dct_benchmark(dlog, a)
# == benchmark_nb(dlog, a)


# In[ ]:

def mk_rnd_dct_arr2d_log(dictsize, nkeys=1000, warmup=None):
    """Generate random dict, select random keys, convert
    dict to log lookup, and access the first element
    to compile the numba method.
    """
    d, a = mk_rnd_dct_arr2d(dictsize, nkeys=nkeys)
    dlog = nmap(d)
    dlog.get(dlog.keys()[0])  # warmup by accessing first key
    if warmup is not None:
        warmup(dlog, a[:2])
    return dlog, a

# def warmup_gen():
#     return mk_rnd_dct_arr2d_log(2, nkeys=1)


# In[ ]:

ts_log_nb = run_benchmark(f=dct_benchmark, genargs=mk_rnd_dct_arr2d_log,
                          inputs=10 ** np.arange(1, 7), n_repeat=10)
ts_log_nb


# warmup = partial(mk_rnd_dct_arr2d_log, warmup=benchmark_nb)
# ts_log_nb_loop = run_benchmark(f=benchmark_nb, genargs=mk_rnd_dct_arr2d_log,
#                           inputs=10 ** np.arange(1, 6), n_repeat=10)
# ts_log_nb_loop

# In[ ]:

times = pd.concat([
    s2df(ts_const, Complexity='Constant'),
    s2df(ts_lin, Complexity='Linear'),
    s2df(ts_lin_nb, Complexity='Linear numba'),
    s2df(ts_log_nb, Complexity='Log'),
#     s2df(ts_log_nb_loop, Complexity='Log_loop'),
])


times_scales = pd.concat([
    s2df(times.set_index('N'), Scale='Linear'),
    s2df(times.set_index('N'), Scale='Log'),
])


# In[ ]:

def plot(x, y, scales, **kw):
    scale, *_ = scales
    p = plt.plot(x, y, '.:')
    if scale == 'Log':
        plt.xscale('log')
    else:
        plt.xticks(rotation=75)
    return p

ts_ = times_scales.query("Complexity == ['Constant', 'Log', 'Log_loop']")
g = sns.FacetGrid(ts_, col='Complexity', row='Scale', sharex=False)
g.map(plot, 'N', 'Time', 'Scale');


# ## Use case

# In[ ]:

@njit
def benchmark_nb(dct, ks):
    s = 0
    for k in ks:
        s += dct.get(k)
    return s


# In[ ]:

# Create dicts and check equivalence
nr.seed(0)
_dict, _ks = mk_rnd_dct_arr2d(10000)
assert type(_dict) == dict


_log_dict = nmap(_dict)

for k in _ks:
    assert _dict.get(k) == _log_dict.get(k)


# In[ ]:

def gen_large_key_sample(dct, size=1000000):
    ks4_ix = nr.choice(np.arange(len(dct)), size=size)
    _ks_all = sorted(dct)
    return [_ks_all[i] for i in ks4_ix]

big_ks = gen_large_key_sample(_dict, size=1000000)


# In[ ]:

f = lambda x: (x ** .3) // 4
fnb = njit(f)

def look_sum_mod(dct, ks):
    return sum([f(dct.get(k))
                for i in range(1, 25)
                for k in ks
                if (k[1] % i) % 2
               ])

def look_sum_loop_mod(dct, ks):
    s = 0
    for i in range(1, 25):
        for k in ks:
            if (k[1] % i) % 2:
                s += fnb(dct.get(k))
    return s

look_sum_loop_mod_nb = njit(look_sum_loop_mod)
look_sum_loop_mod_nb(_log_dict, _ks[:2])


# In[ ]:

get_ipython().magic('time look_sum_loop_mod_nb(_log_dict, big_ks)')


# In[ ]:

get_ipython().magic('time look_sum_mod(_dict, big_ks)')


# In[ ]:

get_ipython().magic('time look_sum_loop_mod(_dict, big_ks)')


# In[ ]:

del outer_loop


# In[ ]:

look_sum_mod(_log_dict, _ks[:2])


# In[ ]:

def look_sum(dct, ks):
#     s = 
    return sum([dct.get(k) for k in ks])

@njit
def look_sum_nb(dct, ks):
    s = 0
    for k in ks:
        s += dct.get(k)
    return s

look_sum_nb(_log_dict, _ks[:2])


# In[ ]:

get_ipython().magic('time look_sum(_dict, _ks)')


# In[ ]:

get_ipython().magic('time look_sum_nb(_log_dict, _ks)')


# In[ ]:

ks4_ix


# In[ ]:

len(_ks)


# In[ ]:

k


# In[ ]:

s_ = 0


# In[ ]:

d


# In[ ]:

_d


# wa = warmup_gen()
# 
# dct_benchmark(*wa)
# 
# dct_benchmark(*warmup_gen())
# 
# 
# %time benchmark_nb(_d, _ks)
# 
# %timeit benchmark_nb(_d, _ks)

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

