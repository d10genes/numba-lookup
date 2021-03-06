from numba_lookup import (
    get_index, tup_dct2arr, sorted_arr_lookup_ix,
    sorted_arr_lookup, nmap2dict, nmap
)
import pytest
from functools import partial

import numpy as np
from scipy import sparse


def lookup_eq_tester(f, dct):
    for (k1, k2), v in dct.items():
        assert f(k1, k2) == v


def coo_todict(cx):
    return {(r, c): v for r, c, v in zip(cx.row, cx.col, cx.data)}


def mk_stuff(rows=12000, cols=1000, random_state=1):
    m_ = sparse.random(rows, cols, density=.05, format='coo',
                       random_state=random_state)
    dct = coo_todict(m_)
    ks, vs = tup_dct2arr(dct)

    mk_dct = pytest.fixture(lambda: dct)
    mk_ks = pytest.fixture(lambda: ks)
    return mk_dct, mk_ks


dct, ks = mk_stuff()


def mk_m(n=12000, m=1000, random_state=1):
    return sparse.random(n, m, density=.05, format='coo',
                         random_state=random_state)


def mk_ks_vs(n=12000, m=1000):
    m = mk_m(n=n, m=m, random_state=1)
    drand = coo_todict(m)
    ks, vs = tup_dct2arr(drand)
    return ks, vs


def test_get_index(ks):
    # unq_vals, unq_ix = np.unique(ks[:, 0], return_index=True)
    unq = get_index(ks)
    unq_vals, unq_ix = unq[:, 0], unq[:, 1]
    assert (ks[:, 0][unq_ix] == unq_vals).all()
    assert (unq_vals[1:] > unq_vals[:-1]).all(), "Should be strictly increasing"
    assert set(unq_vals) == set(ks[:, 0]), 'Should have vals of 1st col'


def test_sorted_arr_lookup_ix(dct):
    ks, vs = tup_dct2arr(dct)
    ix_test = get_index(ks)
    lookup_eq_tester(partial(sorted_arr_lookup_ix, ks, vs, ix_test), dct)


def test_sorted_arr_lookup(dct):
    ks, vs = tup_dct2arr(dct)
    lookup_eq_tester(partial(sorted_arr_lookup, ks, vs), dct)


def test_nm_creation(dct):
    assert nmap2dict(nmap(dct)) == dct


def test_types():
    d1 = {(2, 3): 4., (8, 9): 0}
    lookup_eq_tester(nmap(d1).get, d1)
    d2 = {(2, 3): 4, (8, 9): 0}
    lookup_eq_tester(nmap(d2).get, d2)
    d3 = {(2, 3): np.int64(3), (8, 9): 0}
    lookup_eq_tester(nmap(d3).get, d3)
