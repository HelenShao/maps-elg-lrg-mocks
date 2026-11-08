#!/usr/bin/env python3
"""LRG + the window centrals that the ELG declined but whose halo has an
ELG satellite -- "plusnonELG".

THE QUESTION. The ELG selection draws N objects out of the mass window
logM [11.3, 12.0) and leaves the rest. Those leftovers are in no catalogue
at all: the LRG pool starts at logM >= 12.00, so nothing below that is ever
considered. What does the LRG clustering look like if the leftover CENTRALS
are appended? (Satellites are excluded by default -- --with-satellites
appends them too, which is then the whole window.)

WHAT THIS IS AND IS NOT. The leftovers outnumber the LRGs ~45 to 1 --
11.16M against 340,000 for best_l45 -- so the result is NOT an LRG sample
with a correction, it is the window's centrals with 340,000 massive objects
added, at n ~ 1.15e-2 against the LRG target of 3.4e-4. Scored against Favole
LRG3 the chi2 is therefore dominated by that density and bias change and
does not test agreement with DESI LRGs.

What IS clean is the comparison ACROSS ELG catalogues. The three variants
differ only in which ~0.75-0.97M objects their ELG removed from the window,
so differences between them isolate the imprint of the ELG selection and
nothing else. That is the reason to build all three.

DUPLICATES ARE IMPOSSIBLE HERE, and checked anyway: the LRG pool floor is
logM >= 12.00 and the window is half-open [11.3, 12.0), so no MAP can be in
both. Verified: 0 shared ids.

usage:
  python catalogs/lrg_plusnonelg.py --elg <e.npz> --tag best_l45 --dst <d.npz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

F = Path('/lus/flare/projects/Abacus/helenshao/elg_catalog_runs')
WINDOW = F / 'elg_catalog_base' / 'step0_mass_window.npz'
LRG = (F / 'elg_all_r95_AuroraHighBase_dje' / 'catalogs' / 'lrg_widened'
       / 'lrg_widened_lowdmult_n3p4e4.npz')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--elg', type=Path, required=True)
    ap.add_argument('--tag', required=True,
                    help='name of the ELG catalogue, for the record')
    ap.add_argument('--lrg', type=Path, default=LRG)
    ap.add_argument('--window', type=Path, default=WINDOW)
    ap.add_argument('--all-centrals', action='store_true',
                    help='append EVERY non-ELG window central, not only '
                         'those whose halo hosts an ELG satellite. Off by '
                         'default: that variant adds ~11.2M objects and '
                         'puts the sample at ~34x the LRG target density, '
                         'where chi2 stops testing agreement with DESI.')
    ap.add_argument('--dst', type=Path, required=True)
    a = ap.parse_args(argv)

    w = np.load(a.window, allow_pickle=True)
    l = np.load(a.lrg, allow_pickle=True)
    e = np.load(a.elg, allow_pickle=True)
    wi = np.asarray(w['maps'], np.int64)
    li = np.asarray(l['maps'], np.int64)
    ei = np.asarray(e['maps'], np.int64)
    box = float(w['box'])
    if abs(box - float(l['box'])) > 1e-9:
        raise SystemExit('FATAL: box mismatch, %.3f vs %.3f'
                         % (box, float(l['box'])))
    if abs(float(w['z_cat']) - float(l['z_cat'])) > 1e-9:
        raise SystemExit('FATAL: z_cat mismatch')

    dup = np.intersect1d(wi, li)
    if len(dup):
        raise SystemExit('FATAL: %s MAPs are in BOTH the window and the LRG '
                         '-- appending would double-count them'
                         % f'{len(dup):,}')

    # CENTRALS only. A window satellite that the ELG passed over is still a
    # satellite of some halo; appending those as well would be appending the
    # whole window, and the question asked is about the centrals the ELG
    # selection declined.
    wprim = np.asarray(w['is_halo_primary']).astype(bool)
    wih = np.asarray(w['local_halo_ih'], np.int64)
    eprim = np.asarray(e['is_halo_primary']).astype(bool)
    eih = np.asarray(e['local_halo_ih'], np.int64)

    # The halos where the ELG sample has a SATELLITE but not its central --
    # the conformity complement. Those centrals are in no catalogue at all:
    # the ELG declined them and the LRG pool starts above the window. This
    # asks what happens if the LRG takes them.
    sat_halos = np.unique(eih[~eprim])
    keep = wprim & np.isin(wih, sat_halos) & ~np.isin(wi, ei)
    if a.all_centrals:
        keep = wprim & ~np.isin(wi, ei)
    n_non = int(keep.sum())
    n_sat = int((~eprim).sum())
    print('%s: ELG satellites %s in %s halos; of those halos %s have a '
          'central the ELG did NOT take (%.1f%%)'
          % (a.tag, f'{n_sat:,}', f'{len(sat_halos):,}', f'{n_non:,}',
             100.0 * n_non / max(len(sat_halos), 1)), flush=True)

    # only the per-row fields BOTH files carry; anything else cannot be
    # concatenated without inventing values for one side
    def rowfields(z, n):
        return {k for k in z.files
                if np.asarray(z[k]).ndim >= 1
                and np.asarray(z[k]).shape[0] == n}
    common = sorted(rowfields(w, len(wi)) & rowfields(l, len(li)))
    dropped = sorted((rowfields(w, len(wi)) | rowfields(l, len(li)))
                     - set(common))
    print('   carrying %d per-row fields; dropping %s'
          % (len(common), dropped or 'nothing'), flush=True)

    out = {}
    for k in common:
        out[k] = np.concatenate([np.asarray(l[k]),
                                 np.asarray(w[k])[keep]], axis=0)
    # every row is matched: step0_mass_window excludes unmatched MAPs and
    # the LRG is built from matched rows, so hod_one's gate is satisfied
    out['matched'] = np.ones(len(out['maps']), bool)
    for k in ('Mp', 'box', 'z_cat'):
        out[k] = w[k]

    n = len(out['maps'])
    prim = np.asarray(out['is_halo_primary']).astype(bool)
    out['selection'] = (
        'plusnonELG: %s UNION the mass-window centrals that the ELG '
        'catalogue %s did NOT take AND whose halo hosts at least one ELG '
        'SATELLITE -- the halos where the ELG sample has a satellite but no '
        'central. LRG rows %d + %d such centrals = %d, n = %.4e (%.2fx the '
        'LRG target 3.4e-4). %s Duplicates are impossible: the LRG pool '
        'floor is logM >= 12.00 and the window is half-open below it; '
        'verified 0 shared ids.'
        % (a.lrg.name, a.tag, len(li), n_non, n, n / box ** 3,
           (n / box ** 3) / 3.4e-4,
           'ALL non-ELG centrals were appended (--all-centrals).'
           if a.all_centrals else
           'Restricted to ELG-satellite halos; appending every non-ELG '
           'central instead gives ~34x the target density.'))
    out['elg_partner'] = str(a.elg)
    out['lrg_base'] = str(a.lrg)
    out['n_lrg_base'] = np.int64(len(li))
    out['n_non_elg'] = np.int64(n_non)

    a.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.dst, **out)
    Mp = float(w['Mp'])
    lm = np.log10(np.maximum(np.asarray(out['mult'], np.float64) * Mp, 1e-30))
    print('   wrote %s' % a.dst)
    print('   N=%s  cen=%s  sat=%s  f_sat=%.4f  n=%.4e'
          % (f'{n:,}', f'{int(prim.sum()):,}', f'{int((~prim).sum()):,}',
             1 - prim.mean(), n / box ** 3))
    print('   logM %.3f - %.3f (median %.3f)'
          % (lm.min(), lm.max(), np.median(lm)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
