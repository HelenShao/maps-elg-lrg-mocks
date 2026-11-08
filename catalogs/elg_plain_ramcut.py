#!/usr/bin/env python3
"""The baseline: uniform centrals, lowest ram_int satellites.

Where the project started, and the reference every later rule is measured
against. Centrals thinned uniformly; the satellite slots filled by the lowest
integrated ram_int, on the reading that a satellite stripped of its gas stops
forming stars and stops being an ELG.

Measured w_p 13.45, xi_0 60.93, xi_2 17.82. The inner w_p is the obvious
failure -- r_p*w_p = 19.3 against DESI's 55.3 at r_p = 0.058 -- and chasing
that deficit is what produced every other rule in this directory.

usage:
  python elg_plain_ramcut.py                 # measure, print the chi2
  python elg_plain_ramcut.py --write <x.npz> # also write the catalogue
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from elgmaps_core import (  # noqa: E402
    N_CEN, N_TARGET, SEED, WINDOW, census, load_parent_window, mass_window,
    score, _stratified_rank, _stratified_uniform)

#: written into the catalogue's own `selection`, computed from the arrays
#: rather than inherited -- several catalogues in this project carry their
#: parent's counts and contradict themselves
SELECTION_FMT = ('plain ram_int cut: N=%d = cen %d + sat %d, f_sat=%.4f, '
                 'K_emergent=%.2f (measured, not imposed). Rule: '
                 'reproduce/catalogs/elg_plain_ramcut.py')


def select_plain_ramcut(d, n_cen=N_CEN, n_target=N_TARGET, seed=SEED):
    """The baseline: uniform centrals, lowest ram_int satellites.

    Where the project started. Measured: w_p 13.45, xi_0 60.93, xi_2 17.82.
    """
    w = mass_window(d)
    cen_all = np.flatnonzero(w & d['prim'])
    sat_all = np.flatnonzero(w & ~d['prim'])
    rng = np.random.default_rng(seed)
    cen = cen_all[np.sort(rng.choice(len(cen_all), size=n_cen,
                                     replace=False))]
    o = np.lexsort((d['ids'][sat_all], d['ram'][sat_all]))
    return np.concatenate([cen, sat_all[o[:n_target - len(cen)]]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', type=Path, default=None,
                    help='also write the catalogue here; refuses to '
                         'overwrite an existing file')
    ap.add_argument('--nthreads', type=int, default=32)
    ap.add_argument('--n-cen', type=int, default=N_CEN)
    a = ap.parse_args(argv)
    if a.write is not None and a.write.exists():
        raise SystemExit('FATAL: %s exists; this never overwrites' % a.write)

    d = load_parent_window()
    sel = select_plain_ramcut(d, a.n_cen)
    c = census(d, sel)
    print('plain ram_int cut (the baseline)')
    print('   N=%s cen=%s sat=%s f_sat=%.4f n=%.4e K_emergent=%.2f'
          % (f"{c['N']:,}", f"{c['n_cen']:,}", f"{c['n_sat']:,}",
             c['f_sat'], c['n_density'], c['K_emergent']))
    s = score(d, sel, a.nthreads)
    print('   chi2  wp %.2f  xi0 %.2f  xi2 %.2f   (expect 13.45 / 60.93 / 17.82)'
          % (s['wp'], s['xi0'], s['xi2']))

    if a.write is not None:
        z = np.load(WINDOW, allow_pickle=True)
        n_win = len(d['prim'])
        out = {k: np.asarray(z[k])[sel] for k in z.files
               if np.asarray(z[k]).ndim >= 1
               and np.asarray(z[k]).shape[0] == n_win}
        for k in ('Mp', 'box', 'z_cat'):
            out[k] = z[k]
        out['selection'] = SELECTION_FMT % (c['N'], c['n_cen'], c['n_sat'],
                                            c['f_sat'], c['K_emergent'])
        out['n_cen'] = np.int64(c['n_cen'])
        out['n_sat'] = np.int64(c['n_sat'])
        out['f_sat'] = np.float64(c['f_sat'])
        out['K_emergent'] = np.float64(c['K_emergent'])
        a.write.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.write, **out)
        print('   wrote %s' % a.write)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
