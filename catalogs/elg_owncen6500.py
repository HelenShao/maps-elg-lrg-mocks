#!/usr/bin/env python3
"""owncen6500 -- the first ELG catalogue to recover w_p below r_p = 0.2.

Centrals: a plain UNIFORM thin, no density boost.
Satellites: n_protect of those whose OWN central survived are reserved,
deepest inside the host first; the remaining slots go to the lowest ram_int.

WHY IT EXISTS. Thinning centrals 20:1 leaves only 1.287 per cent of
satellites with their own central, down from 26.163 per cent unthinned. A
plain ram_int cut then keeps 3.54 per cent of satellites, discarding ~96 per
cent of even those, so the one-halo pairs that carry w_p below r_p = 0.2 are
gone before ram_int is consulted at all.

CAVEAT. This catalog is a toy model just to see what satellites are needed to 
recover strong w_p signal at r_p < 100 kpc/h scales. Hence the 6,500 conformally chosen
satellites is FITTED -- scanned from 2,000 to 54,577 against chi2
inside r_p = 0.2 -- and ordering the reserve by smallest r/R_host has no physical justification
e.g. Hadzhiyska+2022 find ELG satellites prefer the OUTSKIRTS. 

Results: measured chi2 values for w_p 1.80, xi_0 36.87, xi_2 3.08.

usage:
  python elg_owncen6500.py                 # measure, print the chi2
  python elg_owncen6500.py --write <x.npz> # also save the catalogue
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
SELECTION_FMT = ('owncen6500: N=%d = cen %d + sat %d, f_sat=%.4f, '
                 'K_emergent=%.2f (measured, not imposed). Rule: '
                 'reproduce/catalogs/elg_owncen6500.py')


def select_owncen6500(d, n_protect=6500, n_cen=N_CEN, n_target=N_TARGET,
                      seed=SEED):
    """The earlier catalogue that first recovered w_p below r_p = 0.2.

    Centrals: a plain UNIFORM thin (no density boost).
    Satellites: `n_protect` whose OWN central survived are reserved, the
    deepest inside the host first; the rest by lowest ram_int.

    Kept for comparison, and honest about its status: 6,500 is a FITTED
    number (scanned 2,000-54,577 against chi2 inside r_p = 0.2) and the
    ordering within the reserve has no published support on its own --
    Hadzhiyska+2022 find ELG satellites prefer the OUTSKIRTS. Measured:
    w_p 1.80, xi_0 36.87, xi_2 3.08.
    """
    w = mass_window(d)
    cen_all = np.flatnonzero(w & d['prim'])
    sat_all = np.flatnonzero(w & ~d['prim'])
    rng = np.random.default_rng(seed)
    cen = cen_all[np.sort(rng.choice(len(cen_all), size=n_cen,
                                     replace=False))]

    own = np.flatnonzero(np.isin(d['ih'][sat_all], np.unique(d['ih'][cen])))
    own = own[np.lexsort((d['ids'][sat_all][own],
                          d['rR'][sat_all][own]))][:n_protect]
    rest = np.setdiff1d(np.arange(len(sat_all)), own)
    rest = rest[np.lexsort((d['ids'][sat_all][rest], d['ram'][sat_all][rest]))]
    n_slot = n_target - len(cen)
    sat = sat_all[np.concatenate([own, rest[:n_slot - len(own)]])]
    return np.concatenate([cen, sat])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', type=Path, default=None,
                    help='also write the catalogue here; refuses to '
                         'overwrite an existing file')
    ap.add_argument('--nthreads', type=int, default=32)
    ap.add_argument('--n-protect', type=int, default=6500)
    ap.add_argument('--n-cen', type=int, default=N_CEN)
    a = ap.parse_args(argv)
    if a.write is not None and a.write.exists():
        raise SystemExit('FATAL: %s exists; this never overwrites' % a.write)

    d = load_parent_window()
    sel = select_owncen6500(d, a.n_protect, a.n_cen)
    c = census(d, sel)
    print('owncen6500')
    print('   N=%s cen=%s sat=%s f_sat=%.4f n=%.4e K_emergent=%.2f'
          % (f"{c['N']:,}", f"{c['n_cen']:,}", f"{c['n_sat']:,}",
             c['f_sat'], c['n_density'], c['K_emergent']))
    s = score(d, sel, a.nthreads)
    print('   chi2  wp %.2f  xi0 %.2f  xi2 %.2f   (expect 1.80 / 36.87 / 3.08)'
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
