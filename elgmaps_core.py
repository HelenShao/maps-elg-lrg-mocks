#!/usr/bin/env python3
"""The shared core: matching, MAP statistics, clustering, scoring.

The per-catalogue selection rules live one per file in catalogs/, each
runnable on its own; this module holds only what more than one of them
needs. Import it directly: `import elgmaps_core as er`, and import each
rule from its own file in catalogs/.

ORIGINAL HEADER FOLLOWS.


The production scripts are spread over a dozen files and two git worktrees,
several of them 400-line batch jobs with their own argument parsers. This
module collects the parts that actually define the catalogues -- the
matching, the two MAP statistics, the three selection rules and the
clustering -- so that a notebook can import one thing and reproduce the
results end to end.

WHAT IS COPIED AND WHAT IS CANONICAL. The definitions below are copies,
kept deliberately short and annotated with where the original lives. If a
number ever disagrees, the production script is right and this file is
stale:

    rsd_shift_z, Cosmology  <- cosmo_params.py          (elg-e1b-v2-pbs)
    wp_dd, xi_dd            <- clustering_elglrg_direct.py   (same)
    RP_EDGES, PIMAX         <- wp_per_catalog.py             (same)
    S_EDGES, N_MU           <- clustering_rocher_hod.py      (same)
    ball matching           <- aurora_analysis.match.ball (Hanyu Zhang)
    ram_int, dmult          <- sidecar_join.py          (lrg-widened-pool)
    the selection rules     <- build_elg_base_satkeep.py, build_elg_censel.py

THE THREE STAGES, and which of them a notebook can actually run:

  Stage 0  MAP -> CompaSO matching over the whole box, and the six-epoch
           sidecar build. BATCH JOBS on a compute node: 121M MAPs against
           the full CompaSO catalogue, and 60 maplog shards decoded twice.
           Hours, not cells. `ball_match` here is the real algorithm and
           runs happily on a subsample, which is how a notebook should
           meet it; `demo_ball_match` does exactly that.
  Stage 1  the mass window and the selection rules. Seconds to minutes,
           from the products of stage 0. This is the part a student runs.
  Stage 2  clustering and figures. Seconds per catalogue at N = 750,000.

usage:
    import sys; sys.path.insert(0, 'catalogs')
    import elgmaps_core as er
    from elg_owncen6500 import select_owncen6500

    cat = er.load_parent_window()
    sel = select_owncen6500(cat)
    rp, wp = er.wp_of(cat, sel)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# 0. WHERE THINGS LIVE
# --------------------------------------------------------------------------
FLARE = Path('/lus/flare/projects/Abacus/helenshao/elg_catalog_runs')
RUN = FLARE / 'elg_all_r95_AuroraHighBase_dje'
CATS = RUN / 'catalogs'

#: every MAP matched to a CompaSO halo at z_cat = 1.1 -- the product of
#: stage 0 and the starting point for everything else (121,386,086 rows)
PARENT = CATS / 'allmaps_compaso_cen_sat.npz'
#: the mass window already applied, 16,320,089 rows; a convenience product
WINDOW = FLARE / 'elg_catalog_base' / 'step0_mass_window.npz'
#: six-epoch histories, split by population; they partition the window
SIDE_CEN = CATS / 'elgbase_sidecar_w11p3_12p0_nall.npz'
SIDE_SAT = CATS / 'elgbase_sidecar_w11p3_12p0_sat.npz'

#: the DESI reference measurements, SHIPPED WITH THIS REPOSITORY so that
#: nothing is read out of a personal home directory. Rocher+2023 Fig. 3
#: (DESI One-Percent Survey ELGs, 0.8 < z < 1.6); the scoring uses the
#: pip+angup columns. Resolved relative to this file, so it works from any
#: working directory.
DESI = Path(__file__).resolve().parent / 'desi'
DESI_TAB = {
    'wp': DESI / 'desi_elg_rpwp_rocher_zenodo.txt',
    'xi0': DESI / 'desi_elg_sxi0_rocher_zenodo.txt',
    'xi2': DESI / 'desi_elg_sxi2_rocher_zenodo.txt',
}

#: the ELG mass window, in MAP mass. NEVER cut inside this -- see the
#: standing instruction in ELG_CATALOG_NEXT_STEPS_PLAN.md section 3.
LOGM_LO, LOGM_HI = 11.3, 12.0
N_TARGET = 750_000            # n = 0.75e-3, read off Rocher's n(z) at z=1.1
N_CEN = 600_000
SEED = 20260911               # the one seed, used by every central thin

# --------------------------------------------------------------------------
# 1. COSMOLOGY AND REDSHIFT SPACE          <- cosmo_params.py
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cosmology:
    H0: float
    Omega_M: float
    Omega_DE: float

    def H_of_z(self, z: float) -> float:
        return self.H0 * ((1.0 + float(z)) ** 3 * self.Omega_M
                          + self.Omega_DE) ** 0.5


#: AbacusAurora HighBase c000. NOT the TNG/Planck-15 cosmology -- using that
#: one biases the RSD shift by ~0.6%, which is why it is named and rejected
#: in cosmo_params.py rather than silently avoided.
HIGHBASE_C000 = Cosmology(H0=67.36, Omega_M=0.3137721, Omega_DE=0.6862279)


def rsd_shift_z(pos, vel, z, cosmo=HIGHBASE_C000):
    """Comoving LOS shift for peculiar v_z (km/s): v_z (1+z) h / H(z).

    Line of sight is +z: these are periodic boxes with no observer in them.
    Positions stay in comoving Mpc/h and the h in the formula is what turns
    a km/s velocity into Mpc/h. The caller wraps back into the box.
    """
    x = np.array(pos, np.float64, copy=True)
    x[:, 2] += (np.asarray(vel, np.float64)[:, 2] * (1.0 + float(z))
                / cosmo.H_of_z(z) * cosmo.H0 / 100.0)
    return x


# --------------------------------------------------------------------------
# 2. STAGE 0 — MATCHING MAPs TO CompaSO HALOES
# --------------------------------------------------------------------------
# The production run is elg_highbase_allmaps_match.py, which streams the
# maplog shards and calls aurora_analysis.match.match_maps_ball. It needs a
# compute node. The function below is the same algorithm, written out so it
# can be read and run on a subsample.
#
# THE HALO CATALOGUE, as recorded in meta/allmaps_cen_sat_match.json:
#
#   compaso_root  /lus/flare/projects/Abacus/helenshao/AbacusSummit/
#                 AbacusSummit_highbase_c000_ph100       (z1.100, 16 slabs)
#   cleaned       True -- abacusnbody CompaSOHaloCatalog(cleaned=True,
#                 subsamples=False), cleaned N == 0 rows dropped; the
#                 cleaning tree is the sibling cleaning/<SimName>/z1.100/
#   halo_R_field  r95_L2com   <-- NOT r100_L2com, which is only the script's
#                 argparse default. The run tag says so: elg_all_r95_*.
#   method/overlap  ball / most_massive
#   n_halo 48,043,269  n_primary 47,654,600
#   121,386,086 MAPs at z_cat, 74,809,159 matched (f = 0.6163)
#
# So the halos are the PUBLIC AbacusSummit HighBase box (3456^3 in 1000
# Mpc/h, Mp = 2.109e9) and the MAPs come from Eisenstein's maplog re-run of
# the same ICs (HighBase_map_nosort, hence the 'dje' tag) -- same phase
# ph100, same cosmology, which is what makes the two joinable at all.
#
# r_over_R in the parent file is still divided by r100_L2com, not by the
# r95 used for matching; run_r95_match.sh:185 states that deliberately.
#
# abacus.par gives Omega_M = 0.315192 against HIGHBASE_C000's 0.3137721
# below: the difference is Omega_Smooth = 0.00142, which the growth
# calculation excludes.

def ball_match(map_pos, halo_pos, halo_R, box, halo_mass,
               radius_factor=1.0, batch_size=100_000, workers=-1):
    """Assign each MAP to a CompaSO halo that CONTAINS it. -1 = unmatched.

    Containment first: a MAP is a candidate for a halo only if it lies
    inside that halo's sphere, d <= radius_factor * R. This is the part
    that differs from a k-nearest match, which always returns a host and
    leaves the caller to apply d < R afterwards.

    When several spheres contain the same MAP the host is the MOST MASSIVE
    of them, ties broken by distance and then halo index. That is the
    CompaSO host-assignment convention.

    Periodicity is handled by cKDTree(boxsize=box) on coordinates folded
    into [0, box), so no manual wrapping is needed.

    Returns (local_ih, n_assigned).
    """
    from scipy.spatial import cKDTree

    map_pos = np.ascontiguousarray(map_pos, np.float64) % box
    halo_pos = np.ascontiguousarray(halo_pos, np.float64) % box
    halo_R = np.ascontiguousarray(halo_R, np.float64) * float(radius_factor)
    halo_mass = np.ascontiguousarray(halo_mass, np.float64)

    # Query from the HALO side: every halo asks which MAPs fall inside it.
    # A tree on the MAPs plus one ball query per halo is cheaper than the
    # reverse, because the halo radii differ and query_ball_point takes a
    # per-point radius.
    tree = cKDTree(map_pos, boxsize=box)
    local_ih = np.full(len(map_pos), -1, np.int64)
    best_mass = np.full(len(map_pos), -np.inf)
    best_d2 = np.full(len(map_pos), np.inf)

    for lo in range(0, len(halo_pos), batch_size):
        hi = min(lo + batch_size, len(halo_pos))
        hits = tree.query_ball_point(halo_pos[lo:hi], r=halo_R[lo:hi],
                                     workers=workers)
        for k, js in enumerate(hits):
            if not js:
                continue
            ih = lo + k
            j = np.asarray(js, np.int64)
            d = map_pos[j] - halo_pos[ih]
            d -= box * np.round(d / box)          # minimum image
            d2 = np.einsum('ij,ij->i', d, d)
            m = halo_mass[ih]
            # most massive wins; equal mass -> closer centre wins
            take = (m > best_mass[j]) | ((m == best_mass[j])
                                         & (d2 < best_d2[j]))
            jt = j[take]
            local_ih[jt] = ih
            best_mass[jt] = m
            best_d2[jt] = d2[take]

    n_assigned = np.bincount(local_ih[local_ih >= 0],
                             minlength=len(halo_pos))
    return local_ih, n_assigned


def central_flags(local_ih, mult):
    """is_halo_primary: the highest-`mult` MAP of THIS SAMPLE per host.

    "Central" is not a property of a MAP on its own, it is a property of a
    MAP *within a sample*: the most massive member of its host. Unmatched
    MAPs (local_ih = -1) are left False. The production catalogues inherit
    this flag and never recompute it, so that a derived catalogue cannot
    silently promote a satellite to a central by dropping its central.
    """
    prim = np.zeros(len(local_ih), bool)
    ok = np.flatnonzero(local_ih >= 0)
    if not len(ok):
        return prim
    # sort by (host, -mult) so the first row of each host is its central
    order = ok[np.lexsort((-np.asarray(mult, np.float64)[ok],
                           np.asarray(local_ih)[ok]))]
    first = np.ones(len(order), bool)
    first[1:] = np.asarray(local_ih)[order][1:] != \
        np.asarray(local_ih)[order][:-1]
    prim[order[first]] = True
    return prim


def demo_ball_match(n_map=200_000, n_halo=20_000, box=200.0, seed=0):
    """Run the matcher on random data, so the algorithm can be seen cheaply.

    Not a test of the catalogues -- a demonstration that the containment
    rule, the most-massive tie-break and the -1 for unmatched behave as
    described, at a size that runs in a notebook cell.
    """
    rng = np.random.default_rng(seed)
    halo_pos = rng.uniform(0, box, (n_halo, 3))
    halo_mass = 10 ** rng.uniform(2, 4, n_halo)
    halo_R = 0.1 * (halo_mass / 1e2) ** (1 / 3)
    # half the MAPs scattered inside haloes, half uniform, so that both the
    # matched and the unmatched branch are exercised
    k = rng.integers(0, n_halo, n_map // 2)
    inside = halo_pos[k] + rng.normal(0, 0.3, (n_map // 2, 3)) * halo_R[k, None]
    map_pos = np.vstack([inside, rng.uniform(0, box, (n_map - len(inside), 3))])
    mult = rng.uniform(20, 500, n_map)
    ih, n_as = ball_match(map_pos, halo_pos, halo_R, box, halo_mass)
    prim = central_flags(ih, mult)
    return {'local_ih': ih, 'n_assigned': n_as, 'is_halo_primary': prim,
            'frac_matched': float((ih >= 0).mean()),
            'n_occupied_hosts': int((n_as > 0).sum()),
            'f_sat': float((~prim[ih >= 0]).mean())}


# --------------------------------------------------------------------------
# 3. THE TWO MAP STATISTICS              <- sidecar_join.py
# --------------------------------------------------------------------------
# Epoch 0 is z_past = 1.2481, the epoch CLOSEST to z_cat = 1.1, so the
# default dmult/mult is the most RECENT interval. The six epochs are
# z = 1.2481, 1.4, 1.5250, 1.6798, 1.8349, 2.0.

def sidecar_fields(ids, is_primary, fields=('mult_ep', 'density_ep',
                                            'rel_vel_ep', 'exists_ep')):
    """Join the six-epoch sidecars on MAP id, population by population.

    Centrals and satellites live in SEPARATE files keyed on the same id
    space, so a row must be looked up in the file matching its own
    is_halo_primary flag. Looking in the wrong file silently returns a
    DIFFERENT object's history, which is why every id must be found or
    this raises.
    """
    out = {}
    for mask, side in ((is_primary, SIDE_CEN), (~is_primary, SIDE_SAT)):
        if not mask.any():
            continue
        s = np.load(side, allow_pickle=True)
        sm = np.asarray(s['maps'], np.int64)
        o = np.argsort(sm, kind='stable')
        sm_s = sm[o]
        want = np.asarray(ids, np.int64)[mask]
        j = np.searchsorted(sm_s, want)
        jj = np.clip(j, 0, len(sm_s) - 1)
        if not ((j < len(sm_s)) & (sm_s[jj] == want)).all():
            raise SystemExit('FATAL: ids missing from %s -- wrong sidecar '
                             'for this population?' % side.name)
        row = o[jj]
        for k in fields:
            v = np.asarray(s[k])[row]
            if k not in out:
                out[k] = np.empty((len(ids),) + v.shape[1:], v.dtype)
            out[k][mask] = v
    return out


def ram_int_of(sc):
    """Integrated ram pressure: sum over epochs of density * rel_vel^2.

    Read as how much stripping the object has suffered over 1.1 < z < 2.
    NOTE for anyone building a selection on it: ram_int correlates +0.667
    with the local density at z_cat and only -0.121 with r/R_host, so
    ranking on it is PARTLY AN INVERSE DENSITY SELECTION. That is the
    single most important thing to know about this variable.
    """
    de = sc['density_ep'].astype(np.float64)
    ve = sc['rel_vel_ep'].astype(np.float64)
    return np.nansum(np.where(sc['exists_ep'], de * ve * ve, 0.0), axis=1)


def dmult_over_mult(sc, mult_now, epoch=0):
    """(mult(z_cat) - mult(z_epoch)) / mult(z_cat) -- an sSFR proxy.

    Epoch 0 is the most recent interval. Gated on exists_ep only, which is
    the convention every other dmult number in the project uses. Do NOT
    also gate on mult_reliable_ep: that flag wants 500 particles in the
    past epoch and a window satellite has 95-474, so it is False for every
    satellite at every epoch and would leave an empty sample.
    """
    m = np.asarray(mult_now, np.float64)
    past = sc['mult_ep'][:, epoch].astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(sc['exists_ep'][:, epoch] & (m > 0),
                        (m - past) / np.where(m > 0, m, 1), np.nan)


# --------------------------------------------------------------------------
# 4. LOADING
# --------------------------------------------------------------------------

def load_parent_window(path=WINDOW, with_sidecar=True):
    """The mass window with the two MAP statistics attached.

    Returns a dict rather than a class so a notebook can see exactly what is
    in it. `prim`, `pos`, `vel` and `ids` are the catalogue; `ram` and
    `dmult` come from the sidecars; `dens`, `relv`, `rR`, `Mh` are the
    z_cat environment carried by the catalogue itself.
    """
    z = np.load(path, allow_pickle=True)
    prim = np.asarray(z['is_halo_primary']).astype(bool)
    d = {'box': float(z['box']), 'z_cat': float(z['z_cat']),
         'Mp': float(z['Mp']), 'prim': prim,
         'pos': np.asarray(z['pos'], np.float64),
         'vel': np.asarray(z['vel'], np.float64),
         'ids': np.asarray(z['maps'], np.int64),
         'ih': np.asarray(z['local_halo_ih'], np.int64),
         'dens': np.asarray(z['density'], np.float64),
         'relv': np.asarray(z['rel_vel'], np.float64),
         'rR': np.asarray(z['r_over_R'], np.float64),
         'Mh': np.asarray(z['M_host'], np.float64),
         'mult': np.asarray(z['mult'], np.float64)}
    d['logM'] = np.log10(np.maximum(d['mult'] * d['Mp'], 1e-30))
    if with_sidecar:
        sc = sidecar_fields(d['ids'], prim)
        d['ram'] = ram_int_of(sc)
        d['dmult'] = dmult_over_mult(sc, d['mult'])
    return d


def mass_window(d, lo=LOGM_LO, hi=LOGM_HI):
    """The step-0 cut. Everything downstream lives inside this."""
    return (d['logM'] >= lo) & (d['logM'] < hi)


# --------------------------------------------------------------------------
# 5. SELECTION PRIMITIVES, shared by the per-catalogue scripts
# --------------------------------------------------------------------------
# The rules themselves now live one per file in catalogs/; these are the
# pieces more than one of them needs. Each rule returns indices into the
# arrays of `d`, and the elg_catalog_base family keeps the central and
# satellite counts fixed, so its f_sat is an input and not a prediction.

def _stratified_uniform(weight, n_keep, rng, tie_ids):
    """n_keep rows, each weight-class sampled UNIFORMLY at its own rate.

    Rows sharing a weight form a class and classes get slots in proportion
    to w * n_class, so the admission RATE is proportional to the weight.
    Inside a class the draw is uniform. This is what keeps the central thin
    a THIN rather than a RANK: ranking centrals on density instead gives
    chi2_wp = 1076, because the 5% most overdense centrals are a wildly
    overbiased sample.
    """
    cls = np.unique(weight)
    idx = [np.flatnonzero(weight == c) for c in cls]
    want = np.array([c * len(i) for c, i in zip(cls, idx)], np.float64)
    want *= n_keep / want.sum()
    cap = np.array([len(i) for i in idx])
    n = np.minimum(np.floor(want).astype(np.int64), cap)
    room = cap - n
    short = n_keep - int(n.sum())
    for j in list(np.argsort(-(want - np.floor(want)))) * 8:
        if short <= 0:
            break
        if room[j] > 0:
            n[j] += 1
            room[j] -= 1
            short -= 1
    if short > 0:
        raise SystemExit('FATAL: %d slots unallocated' % short)
    out = [i if k == len(i)
           else i[np.sort(rng.choice(len(i), size=int(k), replace=False))]
           for i, k in zip(idx, n)]
    return np.sort(np.concatenate(out))


def _stratified_rank(weight, key, tie_ids, n_keep):
    """As above, but the winner inside a class is the smallest key.

    Centrals are drawn uniformly because nothing in the model says which
    central is the better ELG. Satellites DO have such a statement -- the
    least stripped is the one still forming stars -- so theirs is a rank.
    """
    cls = np.unique(weight)
    idx = [np.flatnonzero(weight == c) for c in cls]
    want = np.array([c * len(i) for c, i in zip(cls, idx)], np.float64)
    want *= n_keep / want.sum()
    cap = np.array([len(i) for i in idx])
    n = np.minimum(np.floor(want).astype(np.int64), cap)
    room = cap - n
    short = n_keep - int(n.sum())
    for j in list(np.argsort(-(want - np.floor(want)))) * 8:
        if short <= 0:
            break
        if room[j] > 0:
            n[j] += 1
            room[j] -= 1
            short -= 1
    if short > 0:
        raise SystemExit('FATAL: %d slots unallocated' % short)
    return np.concatenate([i[np.lexsort((tie_ids[i], key[i]))][:int(k)]
                           for i, k in zip(idx, n)])


def emergent_conformity(d, sel):
    """K_emergent: how much a satellite is favoured by its central surviving.

    rate(satellites whose halo hosts a selected central)
      / rate(satellites whose halo does not)

    Measured AFTER the selection and never used to build it. Compare
    Hadzhiyska+2022's kappa = 1.5-1.8 and Gao+2023's K_conf ~ 5; the best
    catalogue predicts 3.28.
    """
    w = mass_window(d)
    sat_all = np.flatnonzero(w & ~d['prim'])
    cen_sel = sel[d['prim'][sel]]
    has = np.isin(d['ih'][sat_all], np.unique(d['ih'][cen_sel]))
    kept = np.isin(sat_all, sel)
    rA = kept[has].mean() if has.any() else 0.0
    rB = kept[~has].mean() if (~has).any() else 0.0
    return float(rA / rB) if rB > 0 else float('inf')


def census(d, sel):
    n_cen = int(d['prim'][sel].sum())
    n_sat = int(len(sel) - n_cen)
    return {'N': int(len(sel)), 'n_cen': n_cen, 'n_sat': n_sat,
            'f_sat': n_sat / float(len(sel)),
            'n_density': len(sel) / d['box'] ** 3,
            'K_emergent': emergent_conformity(d, sel)}


# --------------------------------------------------------------------------
# 6. CLUSTERING            <- clustering_elglrg_direct.py
# --------------------------------------------------------------------------
RP_EDGES = np.logspace(np.log10(0.04), np.log10(32.0), 18)
PIMAX, NPI = 40.0, 40
N_MU = 100
S_EDGES = np.array([
    0.17, 0.20639402, 0.25057936, 0.30422401, 0.36935303, 0.44842504,
    0.54442498, 0.66097682, 0.80248037, 0.97427735, 1.18285305, 1.43608115,
    1.74352094, 2.11677821, 2.56994331, 3.12012311, 3.78808677, 4.59904973,
    5.58362564, 6.77898199, 8.23024317, 9.99219394, 12.13134746,
    14.72845625, 17.8815605, 21.70968908, 26.35735288, 32.0])


def redshift_space(d, sel):
    """Positions in redshift space, wrapped into the box."""
    return np.ascontiguousarray(
        rsd_shift_z(d['pos'][sel], d['vel'][sel], d['z_cat']) % d['box'])


def wp_dd(x1, box, nthreads=32, x2=None):
    """w_p(r_p) = 2 * sum over pi of (DD/RR - 1), with an ANALYTIC RR.

    The randoms are not generated: for a periodic box the expected pair
    count in a cylindrical shell is exact, n1*n2/V * (ring area) * 2 * dpi,
    the 2 being both signs of pi and dpi = 1.
    """
    from Corrfunc.theory.DDrppi import DDrppi
    kw = dict(periodic=True, boxsize=box)
    if x2 is None:
        r = DDrppi(1, nthreads, PIMAX, RP_EDGES, x1[:, 0].copy(),
                   x1[:, 1].copy(), x1[:, 2].copy(), **kw)
        n1 = n2 = len(x1)
    else:
        r = DDrppi(0, nthreads, PIMAX, RP_EDGES, x1[:, 0].copy(),
                   x1[:, 1].copy(), x1[:, 2].copy(), X2=x2[:, 0].copy(),
                   Y2=x2[:, 1].copy(), Z2=x2[:, 2].copy(), **kw)
        n1, n2 = len(x1), len(x2)
    nrp = len(RP_EDGES) - 1
    dd = np.asarray(r['npairs'], np.float64).reshape(nrp, NPI)
    area = np.pi * (RP_EDGES[1:] ** 2 - RP_EDGES[:-1] ** 2)
    rr = (n1 * n2 / box ** 3) * np.outer(area, np.full(NPI, 2.0))
    wp = 2.0 * (dd / rr - 1.0).sum(axis=1)
    return np.sqrt(RP_EDGES[1:] * RP_EDGES[:-1]), wp


def xi_dd(x1, box, nthreads=32, x2=None):
    """s, xi_0, xi_2 from DD/RR - 1 in (s, mu) bins.

    |mu| bins are equal width and solid angle is uniform in mu, so each bin
    holds exactly 1/N_MU of the spherical shell -- there is no geometry
    beyond that. The multipoles are then the Legendre-weighted sums.
    """
    from Corrfunc.theory.DDsmu import DDsmu
    kw = dict(periodic=True, boxsize=box, output_savg=True)
    if x2 is None:
        r = DDsmu(1, nthreads, S_EDGES, 1.0, N_MU, x1[:, 0].copy(),
                  x1[:, 1].copy(), x1[:, 2].copy(), **kw)
        n1 = n2 = len(x1)
    else:
        r = DDsmu(0, nthreads, S_EDGES, 1.0, N_MU, x1[:, 0].copy(),
                  x1[:, 1].copy(), x1[:, 2].copy(), X2=x2[:, 0].copy(),
                  Y2=x2[:, 1].copy(), Z2=x2[:, 2].copy(), **kw)
        n1, n2 = len(x1), len(x2)
    ns = len(S_EDGES) - 1
    dd = np.asarray(r['npairs'], np.float64).reshape(ns, N_MU)
    savg = np.asarray(r['savg'], np.float64).reshape(ns, N_MU)
    vol = (4.0 / 3.0) * np.pi * (S_EDGES[1:] ** 3 - S_EDGES[:-1] ** 3)
    rr = (n1 * n2 / box ** 3) * vol[:, None] * (1.0 / N_MU)
    xi = dd / rr - 1.0

    # s is the PAIR-WEIGHTED mean separation, not the geometric bin centre,
    # falling back to the centre only where a bin is empty. This matters:
    # chi2 interpolates the model onto the DESI s grid, so using the centre
    # instead shifts every point and moved chi2_xi0 from 60.93 to 63.07 when
    # this module was first written. Verified against the production
    # clustering_elglrg_direct.xi_dd, which does the same.
    w = dd.sum(axis=1)
    s = np.where(w > 0, (savg * dd).sum(axis=1) / np.maximum(w, 1),
                 np.sqrt(S_EDGES[1:] * S_EDGES[:-1]))

    # xi_ell = (2l+1)/2 * int_-1^1 xi L_l dmu; for even l with xi even in mu
    # that is (2l+1) * int_0^1, midpoint rule on the folded grid
    mu = (np.arange(N_MU) + 0.5) / N_MU
    dmu = 1.0 / N_MU
    xi0 = (xi * np.ones_like(mu)).sum(axis=1) * dmu
    xi2 = 5.0 * (xi * (0.5 * (3.0 * mu ** 2 - 1.0))).sum(axis=1) * dmu
    return s, xi0, xi2


def wp_of(d, sel, nthreads=32):
    return wp_dd(redshift_space(d, sel), d['box'], nthreads)


def xi_of(d, sel, nthreads=32):
    return xi_dd(redshift_space(d, sel), d['box'], nthreads)


# --------------------------------------------------------------------------
# 7. SCORING AGAINST DESI
# --------------------------------------------------------------------------

def load_desi(stat):
    """x, y, yerr for the pip+angup weighting (Rocher+2023, Zenodo).

    Columns are x, y_pip+angup, yerr_pip+angup, y_No_weight, yerr_No_weight,
    with y already equal to r_p w_p or s xi_l. Errors are symmetric here.
    """
    a = np.loadtxt(DESI_TAB[stat])
    return a[:, 0], a[:, 1], a[:, 2]


def chi2_vs_desi(x, y, stat, smax=None):
    """chi2 PER POINT over the bins the model spans. Not a fit statistic.

    `smax` restricts the multipoles to s < 2, which is the convention used
    throughout this project; w_p is scored over its whole range.
    """
    xd, yp, err = load_desi(stat)
    ok = (np.isfinite(yp) & np.isfinite(err) & (err > 0)
          & (xd >= x.min()) & (xd <= x.max()))
    if smax is not None:
        ok &= xd < smax
    return float(np.sum(((np.interp(xd[ok], x, y) - yp[ok]) / err[ok]) ** 2)
                 / ok.sum())


def score(d, sel, nthreads=32):
    """The three numbers every catalogue in this project is judged by."""
    rp, wp = wp_of(d, sel, nthreads)
    s, xi0, xi2 = xi_of(d, sel, nthreads)
    return {'wp': chi2_vs_desi(rp, rp * wp, 'wp'),
            'xi0': chi2_vs_desi(s, s * xi0, 'xi0', smax=2.0),
            'xi2': chi2_vs_desi(s, s * xi2, 'xi2', smax=2.0),
            'curves': {'rp': rp, 'wp': wp, 's': s, 'xi0': xi0, 'xi2': xi2}}


# --------------------------------------------------------------------------
# 8. ANY CATALOGUE ON DISK
# --------------------------------------------------------------------------
# The rules above rebuild the elg_catalog_base family from the window. Other
# catalogues in this project -- BEST_l45_ramrel003, the LRGs, the
# plusnonELG variants -- come from different lineages whose builders are not
# reimplemented here. These two let any of them be LOADED and MEASURED with
# exactly the same clustering code, so a number can be checked even when the
# build is not reproduced.

def load_catalog(path):
    """A catalogue npz, as the arrays the clustering needs.

    Also returns `selection`, `n_cen`, `n_sat` and `f_sat` BOTH as recorded
    and as measured from the arrays. Keep an eye on the pair: several
    derived catalogues in this project inherited their parent's metadata
    and now disagree with their own contents -- BEST_l45_ramrel003 records
    n_sat = 17,498 and f_sat = 0.0181 while holding 52,576 satellites at
    f_sat = 0.0543. The measured values are the ones to trust.
    """
    z = np.load(Path(path), allow_pickle=True)
    prim = np.asarray(z['is_halo_primary']).astype(bool)
    n = len(prim)
    n_sat = int((~prim).sum())
    out = {'path': str(path), 'box': float(z['box']),
           'z_cat': float(z['z_cat']), 'prim': prim, 'N': n,
           'pos': np.asarray(z['pos'], np.float64),
           'vel': np.asarray(z['vel'], np.float64),
           'ids': np.asarray(z['maps'], np.int64),
           'n_cen_measured': n - n_sat, 'n_sat_measured': n_sat,
           'f_sat_measured': n_sat / float(n),
           'n_density': n / float(z['box']) ** 3,
           'selection': str(z['selection']) if 'selection' in z.files else ''}
    for k in ('n_cen', 'n_sat', 'f_sat'):
        out[k + '_recorded'] = (z[k].item() if k in z.files else None)
    if 'local_halo_ih' in z.files:
        out['ih'] = np.asarray(z['local_halo_ih'], np.int64)
    return out


def score_catalog(path, nthreads=32, elg=True):
    """w_p, xi_0, xi_2 and their chi2 for a catalogue file.

    `elg=False` skips the chi2, because the DESI tables loaded here are the
    Rocher ELG ones; an LRG belongs against Favole+2025 and scoring it on
    these would be meaningless.
    """
    c = load_catalog(path)
    x = np.ascontiguousarray(
        rsd_shift_z(c['pos'], c['vel'], c['z_cat']) % c['box'])
    rp, wp = wp_dd(x, c['box'], nthreads)
    s, xi0, xi2 = xi_dd(x, c['box'], nthreads)
    out = {**{k: v for k, v in c.items() if not hasattr(v, '__len__')
              or isinstance(v, str)},
           'curves': {'rp': rp, 'wp': wp, 's': s, 'xi0': xi0, 'xi2': xi2}}
    if elg:
        out.update({'wp': chi2_vs_desi(rp, rp * wp, 'wp'),
                    'xi0': chi2_vs_desi(s, s * xi0, 'xi0', smax=2.0),
                    'xi2': chi2_vs_desi(s, s * xi2, 'xi2', smax=2.0)})
    return out


#: catalogues this module can LOAD and MEASURE but does NOT rebuild
OTHER_CATALOGUES = {
    'BEST_l45_ramrel003':
        FLARE / 'conformity_r95_z1p10/v2_scan/sat45/BEST_l45_ramrel003'
        / 'catalog.npz',
    'b_l45_cm1185_ri50':
        FLARE / 'conformity_r95_z1p10/v2_scan/b_l45_cm1185_ri50'
        / 'catalog.npz',
    'lrg_widened_lowdmult_n3p4e4':
        RUN / 'catalogs/lrg_widened/lrg_widened_lowdmult_n3p4e4.npz',
    'lrg_sat1278_b+010_ramint':
        FLARE / 'conformity_r95_z1p10/simplified/LRG_catalogs_widened'
        / 'sat1278_b+010_ramint/lrg.npz',
}
