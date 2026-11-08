#!/usr/bin/env python3
"""Rebuild BEST_l45_ramrel003 from scratch, because its record is wrong.

That catalogue's stored `selection` describes its BASE, and its n_cen,
n_sat and f_sat were inherited from the base too -- it records 17,498
satellites and f_sat 0.0181 while holding 52,576 at 0.0543.

The output goes to --out-dir, which must not already hold a catalog.npz, 
and the script refuses to start otherwise (to prevent overwriting).

STAGE 1 -- build_base_v2.py --lam 0.45 --n-cen 950000
                            --sat-cenmass 11.85 --sat-ramint 0.50
  Window elgbase_w11p3_12p0_nall.npz, logM [11.3, 12.0), `matched` applied.

  centrals
    1. candidates: is_halo_primary & matched, with finite dmult/mult
       (sidecar epoch 0, z_past = 1.2481)
    2. rank dmult WITHIN 0.05-dex bins of MAP mass logM, as a
       percentile in [0, 1]. dmult/mult is a specific rate, so at fixed
       absolute accretion a heavier object scores lower; binning removes
       that artefact.
    3. score = dmult/mult + lam * (logM - 11.3) / 0.7 - model of the [OII] flux limit, 
       which rises with SFR (dmult) and with mass
    4. take the top 950,000, ties broken by higher mass then higher dmult.

  satellites (STRICT conformity)
    5. same halo as a selected central, matched, not primary
    6. dmult/mult >= 0.10 -- the same ELG-ness test the centrals passed
    7. drop those whose ELG central has logM > 11.85 (sat_cenmass)
    8. drop the top 50% by integrated ram_int (sat_ramint), the percentile
       being taken AFTER step 7
  -> 950,000 + 17,498 = 967,498, conformity 100%

STAGE 2 -- build_minimal_conformity.py --base <stage 1> --ramrel 0.03
  we want some of the satellites that strict conformity threw away, so match
    satellites whose halo holds NO elg central, but still passes
    dmult >= 0.10 criteria.
  lm_cen: the halo's primary from allmaps_compaso_primaries.npz, NOT the
    window -- these halos often have a central far above logM 12.0, which
    is why they were never conformal candidates.
  ramrel(0.03): within 0.2-dex bins of lm_cen holding >= 20 members, keep
    ram_int <= the 3rd percentile OF THAT BIN. Raw ram_int tracks halo
    density, so an unbinned cut would just select low-mass haloes; ranking
    within central-mass bins leaves "how long has THIS satellite been
    exposed, compared with peers in a comparable halo".
  trim_to_abundance: raise a threshold on the base's cen_score until
    kept centrals + their surviving conformal satellites + the admitted
    pool == 967,498. Removing a central orphans its satellites, so the
    total is re-evaluated in full at each trial threshold (bisection).
  -> 914,922 + 16,775 + 35,801 = 967,498, conformity 31.9%

usage:
  python catalogs/elg_best_l45.py --out-dir <a NEW dir> [--verify-against <orig>]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

CATS = Path('/lus/flare/projects/Abacus/helenshao/elg_catalog_runs'
            '/elg_all_r95_AuroraHighBase_dje/catalogs')

# all central MAPs as identified by compaso
PRIM = CATS / 'allmaps_compaso_primaries.npz' 
# all maps in mass window [11.3, 12.0)
WIN = CATS / 'elgbase_w11p3_12p0_nall.npz' 

# these files contain MAP data across 6 maplog epochs, 
# called "sidecars" cuz they are additional maplog quantities appended to the main catalog
SIDE_CEN = CATS / 'elgbase_sidecar_w11p3_12p0_nall.npz'
SIDE_SAT = CATS / 'elgbase_sidecar_w11p3_12p0_sat.npz'

EPOCH = 0
LO, HI = 11.3, 12.0
OUT_KEYS = ('maps', 'pos', 'vel', 'mult', 'local_halo_ih', 'is_halo_primary',
            'matched', 'M_host', 'R_host', 'r_over_R', 'dmult_over_mult')


def _dmult(ids, side, mult_now):
    """dmult/mult against the sidecar's epoch 0 (z_past = 1.2481)."""
    sm = np.asarray(side['maps'], np.int64)
    ii = np.searchsorted(sm, ids)
    ok = (ii < len(sm)) & (sm[np.minimum(ii, len(sm) - 1)] == ids)
    if not ok.all():
        raise SystemExit('FATAL: %d of %d absent from the sidecar'
                         % (int((~ok).sum()), len(ids)))
    past = np.asarray(side['mult_ep'])[ii, EPOCH].astype(np.float64)
    ex = np.asarray(side['exists_ep'])[ii, EPOCH].astype(bool)
    return np.where(ex & (mult_now > 0),
                    (mult_now - past) / np.maximum(mult_now, 1e-30), np.nan)


def _ram_int(ids, side):
    sm = np.asarray(side['maps'], np.int64)
    ii = np.searchsorted(sm, ids)
    ok = (ii < len(sm)) & (sm[np.minimum(ii, len(sm) - 1)] == ids)
    if not ok.all():
        raise SystemExit('FATAL: %d ids absent from the satellite sidecar'
                         % int((~ok).sum()))
    ex = np.asarray(side['exists_ep'])[ii].astype(bool)
    de = np.asarray(side['density_ep'])[ii].astype(np.float64)
    ve = np.asarray(side['rel_vel_ep'])[ii].astype(np.float64)
    return np.nansum(np.where(ex, de * ve * ve, 0.0), axis=1)


def stage1(lam=0.45, n_cen=950_000, sat_dmult=0.10, sat_cenmass=11.85,
           sat_ramint=0.50):
    """The base: scored centrals plus strictly conformal satellites."""
    w = np.load(WIN, allow_pickle=True)
    prim = np.asarray(w['is_halo_primary']).astype(bool)
    match = np.asarray(w['matched']).astype(bool)
    mult = np.asarray(w['mult'], np.float64)
    ids = np.asarray(w['maps'], np.int64)
    ih = np.asarray(w['local_halo_ih'], np.int64)
    Mp = float(w['Mp'])

    pc = np.flatnonzero(prim & match)
    d = _dmult(ids[pc], np.load(SIDE_CEN, allow_pickle=True), mult[pc])
    good = np.isfinite(d)
    pc, d = pc[good], d[good]
    lm = np.log10(np.maximum(mult[pc] * Mp, 1e-30))

    edges = np.arange(LO, HI + 0.001, 0.05)
    b = np.clip(np.digitize(lm, edges), 1, len(edges) - 1)
    pct = np.zeros(len(d))
    for k in np.unique(b):
        m = b == k
        pct[m] = np.argsort(np.argsort(d[m])) / float(max(m.sum() - 1, 1))
    score = pct + lam * (lm - LO) / (HI - LO)
    order = np.lexsort((-d, -lm, -score))[:n_cen]
    cen = pc[order]
    print('   stage 1 centrals: %s' % f'{len(cen):,}', flush=True)

    host = np.unique(ih[cen])
    sat = (~prim) & match & np.isin(ih, host)
    si = np.flatnonzero(sat)
    ss = np.load(SIDE_SAT, allow_pickle=True)
    ds = _dmult(ids[si], ss, mult[si])
    keep = np.isfinite(ds) & (ds >= sat_dmult)
    si, ds = si[keep], ds[keep]
    print('   conformal satellites, dmult >= %.2f: %s'
          % (sat_dmult, f'{len(si):,}'), flush=True)

    if sat_cenmass > 0:
        cen_of = dict(zip(ih[cen].tolist(), np.arange(len(cen)).tolist()))
        rr = np.array([cen_of.get(int(h), -1) for h in ih[si]], np.int64)
        lm_cen_of_sat = np.where(rr >= 0, lm[order][np.maximum(rr, 0)], np.nan)
        k = np.isfinite(lm_cen_of_sat) & (lm_cen_of_sat <= sat_cenmass)
        si, ds = si[k], ds[k]
        print('   after central logM <= %.2f: %s'
              % (sat_cenmass, f'{len(si):,}'), flush=True)
    if sat_ramint > 0:
        ri = _ram_int(ids[si], ss)
        thr_r = np.percentile(ri, 100.0 * (1.0 - sat_ramint))
        k = ri < thr_r
        si, ds = si[k], ds[k]
        print('   after dropping the top %.0f%% by ram_int: %s'
              % (100 * sat_ramint, f'{len(si):,}'), flush=True)

    rows = np.concatenate([cen, si])
    out = {k: np.asarray(w[k])[rows] for k in OUT_KEYS
           if k != 'dmult_over_mult'}
    out['dmult_over_mult'] = np.concatenate([d[order], ds])
    out['cen_score'] = np.concatenate([score[order],
                                       np.full(len(si), -np.inf)])
    return w, out, len(cen), len(si)


def stage2(w, base, ramrel_t=0.03, sat_dmult=0.10):
    """The ramrel overlay: admit from the pool, trim the base to hold N."""
    prim = np.asarray(w['is_halo_primary']).astype(bool)
    match = np.asarray(w['matched']).astype(bool)
    ih = np.asarray(w['local_halo_ih'], np.int64)
    ids = np.asarray(w['maps'], np.int64)
    mult = np.asarray(w['mult'], np.float64)
    Mp = float(w['Mp'])

    si = np.flatnonzero((~prim) & match)
    b_prim = np.asarray(base['is_halo_primary']).astype(bool)
    host_elg = np.unique(np.asarray(base['local_halo_ih'], np.int64)[b_prim])
    si = si[~np.isin(ih[si], host_elg)]
    print('   pool (halo has no base central): %s' % f'{len(si):,}',
          flush=True)

    ss = np.load(SIDE_SAT, allow_pickle=True)
    dmu = _dmult(ids[si], ss, mult[si])
    k = np.isfinite(dmu) & (dmu >= sat_dmult)
    si, dmu = si[k], dmu[k]
    print('   after dmult >= %.2f: %s' % (sat_dmult, f'{len(si):,}'),
          flush=True)
    ram_int = _ram_int(ids[si], ss)

    # the halo's central from the ALL-HALO primaries, not the window: these
    # halos often hold a central far above logM 12.0
    p = np.load(PRIM, allow_pickle=True)
    pih = np.asarray(p['local_halo_ih'], np.int64)
    o = np.argsort(pih)
    pih_s, pmult_s = pih[o], np.asarray(p['mult'], np.float64)[o]
    del p, pih, o
    kk = np.searchsorted(pih_s, ih[si])
    okk = (kk < len(pih_s)) & (pih_s[np.minimum(kk, len(pih_s) - 1)] == ih[si])
    if not okk.all():
        raise SystemExit('FATAL: %d pool satellites have no halo primary'
                         % int((~okk).sum()))
    lm_cen = np.log10(np.maximum(pmult_s[kk] * Mp, 1e-30))

    edges = np.arange(np.floor(np.nanmin(lm_cen) * 10) / 10,
                      np.nanmax(lm_cen) + 0.2, 0.2)
    admit = np.zeros(len(lm_cen), bool)
    bb = np.digitize(lm_cen, edges)
    for key in np.unique(bb):
        m = bb == key
        if m.sum() < 20:
            continue
        admit[m] = ram_int[m] <= np.percentile(ram_int[m], 100.0 * ramrel_t)
    n_add = int(admit.sum())
    print('   ramrel %.2f admits: %s' % (ramrel_t, f'{n_add:,}'), flush=True)

    # trim the base so that kept centrals + their surviving conformal
    # satellites + the admitted pool lands exactly on the base's N
    n_target = len(np.asarray(base['maps']))
    d = np.asarray(base['cen_score'], np.float64)
    bih = np.asarray(base['local_halo_ih'], np.int64)
    ci, bsi = np.flatnonzero(b_prim), np.flatnonzero(~b_prim)
    cand = np.unique(d[ci][np.isfinite(d[ci])])

    def total(thr):
        keep_c = ci[np.isfinite(d[ci]) & (d[ci] >= thr)]
        hosts = np.unique(bih[keep_c])
        keep_s = bsi[np.isin(bih[bsi], hosts)]
        return len(keep_c) + len(keep_s) + n_add, keep_c, keep_s

    if total(cand[0])[0] <= n_target:
        thr = cand[0]
    else:
        lo, hi = 0, len(cand) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if total(cand[mid])[0] > n_target:
                lo = mid + 1
            else:
                hi = mid
        thr = cand[lo]
    n_tot, keep_c, keep_s = total(thr)
    print('   central threshold %.10f -> %s cen + %s conformal sat + %s '
          'admitted = %s' % (thr, f'{len(keep_c):,}', f'{len(keep_s):,}',
                             f'{n_add:,}', f'{n_tot:,}'), flush=True)

    rows_base = np.concatenate([keep_c, keep_s])
    add_idx = si[admit]
    out = {}
    for k in OUT_KEYS:
        if k == 'dmult_over_mult':
            out[k] = np.concatenate([
                np.asarray(base[k], np.float64)[rows_base], dmu[admit]])
        else:
            out[k] = np.concatenate([np.asarray(base[k])[rows_base],
                                     np.asarray(w[k])[add_idx]])
    return out, thr, n_add, len(keep_c), len(keep_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--lam', type=float, default=0.45)
    ap.add_argument('--n-cen', type=int, default=950_000)
    ap.add_argument('--sat-cenmass', type=float, default=11.85)
    ap.add_argument('--sat-ramint', type=float, default=0.50)
    ap.add_argument('--ramrel', type=float, default=0.03)
    ap.add_argument('--verify-against', type=Path, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()

    base_p = a.out_dir / 'base_catalog.npz'
    out_p = a.out_dir / 'catalog.npz'
    for p in (base_p, out_p):
        if p.exists():
            raise SystemExit('FATAL: %s already exists. This script never '
                             'overwrites; choose an empty --out-dir.' % p)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print('STAGE 1  base (lam %.2f, n_cen %s, cenmass %.2f, ramint %.2f)'
          % (a.lam, f'{a.n_cen:,}', a.sat_cenmass, a.sat_ramint), flush=True)
    w, base, n_c1, n_s1 = stage1(a.lam, a.n_cen, 0.10, a.sat_cenmass,
                                 a.sat_ramint)
    meta = dict(Mp=w['Mp'], box=w['box'], z_cat=w['z_cat'])
    np.savez_compressed(base_p, **base, **meta, n_cen=n_c1, n_sat=n_s1,
                        f_sat=n_s1 / float(n_c1 + n_s1), lam=a.lam)
    print('   wrote %s' % base_p.name, flush=True)

    print('STAGE 2  ramrel %.2f overlay' % a.ramrel, flush=True)
    base_z = np.load(base_p, allow_pickle=True)
    out, thr, n_add, n_c2, n_s2 = stage2(w, base_z, a.ramrel)

    n = len(out['maps'])
    n_sat = int((~np.asarray(out['is_halo_primary']).astype(bool)).sum())
    # metadata computed from THIS file, which is the bug being fixed
    np.savez_compressed(
        out_p, **out, **meta,
        n_cen=np.int64(n - n_sat), n_sat=np.int64(n_sat),
        f_sat=np.float64(n_sat / float(n)),
        minconf_rule=json.dumps({'rule': 'ramrel', 'value': a.ramrel}),
        minconf_base=str(base_p), minconf_cen_thr=np.float64(thr),
        minconf_n_added=np.int64(n_add),
        selection=(
            'Stage 1: centrals = top '
            '%d by [dmult percentile within 0.05-dex logM bins] + %.2f x '
            '(logM-11.3)/0.7, ties on higher mass then higher dmult; '
            'satellites = same halo as a selected central (STRICT '
            'conformity), dmult >= 0.10, central logM <= %.2f, then the top '
            '%.0f%% by ram_int dropped. Stage 2: %d satellites admitted from '
            'the NON-conformal pool (halo holds no stage-1 central, dmult >= '
            '0.10) by ramrel <= %.2f, i.e. ram_int below that percentile '
            'WITHIN 0.2-dex bins of the halo central mass from the all-halo '
            'primaries; the base was then trimmed at cen_score >= %.10f to '
            'hold N. Result %d = %d centrals + %d satellites, f_sat %.4f. '
            'Unlike the original, every field here is computed from this '
            'file.'
            % (a.n_cen, a.lam, a.sat_cenmass, 100 * a.sat_ramint, n_add,
               a.ramrel, thr, n, n - n_sat, n_sat, n_sat / float(n))))
    print('   wrote %s' % out_p.name, flush=True)
    print('REBUILT  N=%s  cen=%s  sat=%s  f_sat=%.4f   [%.1f min]'
          % (f'{n:,}', f'{n - n_sat:,}', f'{n_sat:,}', n_sat / float(n),
             (time.time() - t0) / 60), flush=True)

    if a.verify_against:
        o = np.load(a.verify_against, allow_pickle=True)
        oi = np.asarray(o['maps'], np.int64)
        ni = np.asarray(out['maps'], np.int64)
        op = np.asarray(o['is_halo_primary']).astype(bool)
        same = len(oi) == len(ni) and np.array_equal(np.sort(oi), np.sort(ni))
        print('\nVERIFY against %s' % a.verify_against)
        print('   N            %s vs %s' % (f'{len(ni):,}', f'{len(oi):,}'))
        print('   centrals     %s vs %s'
              % (f'{n - n_sat:,}', f'{int(op.sum()):,}'))
        print('   satellites   %s vs %s'
              % (f'{n_sat:,}', f'{int((~op).sum()):,}'))
        print('   identical MAP id set: %s' % same)
        if not same:
            only_new = np.setdiff1d(ni, oi)
            only_old = np.setdiff1d(oi, ni)
            print('   only in rebuild %s, only in original %s'
                  % (f'{len(only_new):,}', f'{len(only_old):,}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
