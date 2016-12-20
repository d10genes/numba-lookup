
# coding: utf-8

# I recently had occasion to use dictionary or sparse matrix to keep track of co-occurrence counts, neither of which I could use with Numba. This isn't the first time I've seen this limitation, so I thought it would be worth my time to look for the best solution I could conjure with the minimal amount of energy. 
# 
# While python's arrays and dicts come with the convenience of constant time lookups, I couldn't think of a straightforward strategy to use these in Numba. 

# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')

from project_imports import *


# In[ ]:

import scipy as sp
import seaborn as sns
from numba import njit

from pandas.compat import lmap, lfilter, lrange, lzip
import numba_lookup as nl

from tests.test_lookup import *


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "Jupyter.keyboard_manager.command_shortcuts.add_shortcuts({\n'Ctrl-k': 'jupyter-notebook.move-selected-cell-up',\n'Ctrl-j': 'jupyter-notebook.move-selected-cell-down',\n'Shift-m': 'jupyter-notebook.merge-selected-cell-with-cell-after'})")


# # Benchmarks: constant vs linear vs log
# 
# Here is some code that benchmarks different lookup strategies to compare how much slower they get as the number of elements in the lookup structure increases. The function `run_benchmark` takes a function `f` that does a lookup, and function `genargs(n)` that creates the lookup data structure with `n` elements. Since we want to see how the size affects the lookup speed, different input sizes can be passed through the `inputs` arg, and the function will return a Series with the average lookup time for each size.

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
# 
# To illustrate the ideal performance I'd like, we'll first benchmark constant time lookups. The functions here will randomly generate dictionaries of size `n` with 2-tuple int keys and int values (i.e., `{(2, 3): 4}`), in the form that a co-occurrence lookup would have. It will then generate a random subset of keys, which the benchmark function `dct_benchmark` will use to look up corresponding values and sum. 

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


# Since we've left a computer science textbook and entered a computer, there will be some variation. But the general trend here is that, even when the size of the dict increases 10x, this does not have a 10x increase in the lookup time (as a linear time lookup would), but rather remains about the same. 

# ## Linear lookup

# As nice as python's built-in hash tables are, they haven't been ported to numba yet, so we can't really take advantage of them in large for-loops.
# 
# Though it took me longer than I'd be comfortable admitting, there's a simple solution to get around dict usage that can use arrays. Arranging the 2 keys and 1 value as 3 columns in an array in this form:

# In[ ]:

DataFrame([[3, 4, 1], [2, 3, 4]], columns=['Key 1', 'Key 2', 'Value'])


# we can look up the co-occurrence count of indices `2` and `3`, for example, by scanning through the first and second columns until we see the values 2 and 3, and then return the 3rd entry in that row (in this case, `4`).

# In[ ]:

class DWrapper:
    """Wrap an array and lookup function to use
    same api as a dict."""
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


# The times here show that as the size of the array increases by an order of magnitude, the time it takes to look up a single element also increases by (very roughly) an order of magnitude. While dicts are relatively unscathed by bigger sizes, a linear speed lookup like the array above will make lookups impossibly slow, particularly with a long loop.

# ### Numba speedup

# To show how devastating linear complexity is for large N, here is a similar lookup array that uses numba to speed things up. 

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


# While the time it takes to scan through an array is about an order of magnitude slower than the time it takes in pure python, you see the same explosion in time as the size of the array increases. Numba buys you about an order of magnitude in time with a linear lookup before it also becomes unusable for large loops.
# 
# The charts below show the times in both log and linear scale. The linear scale is probably better here, and shows the constant uptick in time it takes to retrieve from the linear lookups, but with a lower slope for the lookups done in numba functions. 

# In[ ]:

# Tabulate times for easy plotting
def tab_time_scales(xs):
    times = pd.concat(xs)
    return pd.concat([
        s2df(times.set_index('N'), Scale='Linear'),
        s2df(times.set_index('N'), Scale='Log'),
    ])
    
def plot(x, y, scales, **kw):
    scale, *_ = scales
    p = plt.plot(x, y, '.:')
    if scale == 'Log':
        plt.xscale('log')
    else:
        plt.xticks(rotation=75)
    return p

times_scales = tab_time_scales([
    s2df(ts_const, Complexity='Constant'),
    s2df(ts_lin, Complexity='Linear'),
    s2df(ts_lin_nb, Complexity='Linear numba'),
])


# In[ ]:

g = sns.FacetGrid(times_scales, col='Complexity', row='Scale', sharex=False)
g.map(plot, 'N', 'Time', 'Scale');


# 
# times = pd.concat([
#     s2df(ts_const, Complexity='Constant'),
#     s2df(ts_lin, Complexity='Linear'),
#     s2df(ts_lin_nb, Complexity='Linear numba'),
# ])
# 
# 
# times_scales = pd.concat([
#     s2df(times.set_index('N'), Scale='Linear'),
#     s2df(times.set_index('N'), Scale='Log'),
# ])
# 
# times_scales = pd.concat([
#     s2df(times.set_index('N'), Scale='Linear'),
#     s2df(times.set_index('N'), Scale='Log'),
# ])

# ## Logarithmic time lookup
# 
# The intuition for log-speed lookups can be understood when searching for a word in the dictionary. (The old fashioned kind, that is. Back when you had to turn pages made of paper.) We don't have the page number for each word memorized, so we can't instantly flip to a word's page (which would be a constant time operation, no matter how large the dictionary is).
# 
# 
# But we also don't use dictionaries where the words are randomly shuffled, requiring us to scan through from the beginning, page by page, word by word until we find our target entry (aka linear lookup complexity).
# 
# The words are sorted, so at worst case we can turn to the middle of the dictionary, see that our word comes in the second half, and repeat this search in the second half of the book, eliminating half of the remaining pages at each step. If a 1000 page dictionary is doubled to 2000 pages, the lookup won't take twice as long (linear increase), it'll just be an extra step (log increase). This illustrates a log time lookup, and only requires us to sort our array and use a recursive search strategy. 
# 
# ### Benchmark Log lookup
# 
# Here I'm importing a wrapper `nmap` from the `numba_lookup.py` file that's like `DWrapper` above. This essential difference is that this function sorts the array after it is converted from the input dict, and uses numpy's (and therefore numba's) `np.searchsorted` functionality to look up the keys in the first two columns.

# In[ ]:

from numba_lookup import nmap


# In[ ]:

dlog = nmap(d)

# Check equivalence
assert dct_benchmark(d, a) == dct_benchmark(dlog, a)


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


# In the benchmark we would expect that for each time the size's order of magnitude increases, the time it takes to retrieve an element will increase by a constant amount. This is (again, *very roughly*) what we see in the log scale plot:

# In[ ]:

times_scales2 = tab_time_scales([
    s2df(ts_const, Complexity='Constant'),
    s2df(ts_lin, Complexity='Linear'),
    s2df(ts_lin_nb, Complexity='Linear numba'),
    s2df(ts_log_nb, Complexity='Log'),
])

ts_ = times_scales2.query("Complexity == ['Constant', 'Log', 'Log_loop']")
g = sns.FacetGrid(ts_, col='Complexity', row='Scale', sharex=False)
g.map(plot, 'N', 'Time', 'Scale');


# ## Use case

# Since logarithmic time lookups increase at a faster rate than constant time operations, justification is still needed for why I care so much about these numba lookup structures when the built-in dicts have better asymptotic complexity. To answer this, here are some benchmarks that are closer to my use case, in that the lookups are performed within a larger (and nested) loop, with a simple function applied in each loop. While the lookup was the core of the other benchmarks, it makes up a smaller fraction of each loop in this benchmark.

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
look_sum_loop_mod_nb(_log_dict, _ks[:2]);


# In[ ]:

get_ipython().magic('time res1 = look_sum_loop_mod_nb(_log_dict, big_ks)')


# In[ ]:

get_ipython().magic('time res2 = look_sum_mod(_dict, big_ks)')


# In[ ]:

get_ipython().magic('time res3 = look_sum_loop_mod(_dict, big_ks)')


# In[ ]:

assert res1 == res2 == res3


# Because of the low cost of each iteration in a numba loop, nested loops like this are where numba really shines. Additional factors such as the overhead of entering and exiting a loop and doing basic operations in native python is so high that as the number of iterations increases, the performance of a single lookup, even with better runtime complexity, matters less and less.
# 
# Thus, with just some basic knowledge of algorithmic complexity, we get speedy lookups using already-implemented numba functions at a pretty low cost.
