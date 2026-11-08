# maps-elg-lrg-mocks

Mock **ELG** and **LRG** catalogues at *z* = 1.1, built from Aurora MAPs
matched to CompaSO halos and tuned against DESI clustering — Rocher+2023
(ELG One-Percent) for the ELGs, Favole+2025 (Y1 LRG3 and ELG1×LRG3) for the
LRGs.

> These scripts read data products that live on Aurora's Flare filesystem.
> The paths are hard-coded. Off that filesystem the scripts will raise
> `FileNotFoundError` at the first `np.load`; the recipes are still readable,
> but nothing will run without those files.

## The files

| file | what it is |
|---|---|
| `elgmaps_core.py` | shared machinery: MAP→halo ball matching, the six-epoch sidecar statistics (`ram_int`, `dmult`), the mass-window loader, `wp_dd` / `xi_dd`, and the DESI scoring |
| `catalogs/elg_plain_ramcut.py` | the baseline ELG rule |
| `catalogs/elg_owncen6500.py` | the own-central reserve ELG rule |
| `catalogs/elg_best_l45.py` | the two-stage conformity rebuild |
| `catalogs/lrg_plusnonelg.py` | the LRG variant |

Each script runs on its own and prints its χ²:

```bash
python catalogs/elg_owncen6500.py                  # measure only
python catalogs/elg_owncen6500.py --write new.npz  # also write; never overwrites
```

Every script refuses to overwrite an existing output file.

## The rules

χ² is **per point** against DESI, with the multipoles taken over *s* < 2.
These are not fit statistics — they are how catalogues are ranked here.

### `elg_plain_ramcut.py` — the baseline  ·  w_p 13.45 · ξ₀ 60.93 · ξ₂ 17.82

Centrals thinned uniformly; the satellite slots filled by the lowest
integrated ram pressure, on the reading that a satellite stripped of its gas
stops forming stars and stops being an ELG.

Its inner `w_p` is the obvious failure — r_p·w_p = 19.3 against DESI's 55.3
at r_p = 0.058 — and chasing that deficit is what produced every other rule
here.

### `elg_owncen6500.py` — the own-central reserve  ·  w_p 1.80 · ξ₀ 36.87 · ξ₂ 3.08

Centrals thinned uniformly, no density boost. Then `n_protect = 6500`
satellites whose **own** central survived the thinning are reserved, deepest
inside the host first; the remaining slots go to the lowest `ram_int`.

Why it is needed: thinning centrals 20:1 leaves only 1.287% of satellites
with their own central, down from 26.163% unthinned. A plain `ram_int` cut
then keeps 3.54% of satellites, discarding ~96% of even those — so the
one-halo pairs that carry `w_p` below r_p = 0.2 are gone before `ram_int` is
consulted at all.

**Honest status.** 6,500 is a **fitted** number, scanned from 2,000 to 54,577
against χ² inside r_p = 0.2. Ordering the reserve by smallest `r/R_host` has
no published support on its own: Hadzhiyska+2022 find ELG satellites prefer
the *outskirts*. It remains the best `w_p` here.

### `elg_best_l45.py` — the two-stage conformity rebuild

The only script here that rebuilds an **existing** catalogue rather than
defining a rule, because that catalogue's stored recipe describes its *base*
and so cannot document itself. It reproduces it exactly:

```
VERIFY  N 967,498 vs 967,498 | cen 914,922 vs 914,922
        sat 52,576 vs 52,576 | identical MAP id set: True
```

~15 minutes.

**Stage 1** — centrals: rank `dmult` as a percentile *within 0.05-dex logM
bins* (`dmult` is a specific rate, so a global threshold would merely keep
light galaxies), then `score = pctile + 0.45·(logM − 11.3)/0.7`, the mass
tilt standing in for the [OII] flux limit. Satellites: **strict conformity** —
same halo as a selected central, `dmult ≥ 0.10`, central `logM ≤ 11.85`,
then drop the top 50% by `ram_int`. Result: 950,000 + 17,498, **100%
conformal**.

**Stage 2** — the pool is the satellites that strict conformity *threw away*:
halos holding **no** stage-1 central. Within 0.2-dex bins of the host
central's mass, keep `ram_int ≤ 3rd percentile of that bin`; then trim the
base by bisection until the total returns to 967,498. Result: 914,922 +
16,775 + 35,801, **31.9% conformal**.

So the final catalogue's 35,801 stage-2 satellites are non-conformal **by
construction** — the pool is *defined* as halos with no base central. "Same
halo" describes stage 1 only.

Unlike the original, the rebuilt file's `n_cen` / `n_sat` / `f_sat` are
computed from its own arrays rather than inherited.

### `lrg_plusnonelg.py` — the LRG variant

Appends to `lrg_widened_lowdmult_n3p4e4` the non-ELG centrals in the ELG mass
window [11.3, 12.0) **whose halo hosts an ELG satellite**.

With `elg_best_l45` it appends 8,390 centrals and improves the LRG:

| | before | after |
|---|---|---|
| LRG auto `w_p` χ² | 3.61 | **0.93** |
| ELG×LRG cross χ² | 18.46 | **15.06** |

It is **destructive for other ELG catalogues**, which have far more ELG
satellites sitting around non-ELG centrals: appending 83,792 or 117,181
centrals instead drives the LRG auto `w_p` χ² to 81 and 97. Check how many
centrals it appends before trusting the result.

`--all-centrals` appends every non-ELG window central instead, which adds
~11.2M objects; it is off by default.

## Data products these scripts read

All under `/lus/flare/projects/Abacus/helenshao/elg_catalog_runs/`:

| file | what |
|---|---|
| `elg_all_r95_AuroraHighBase_dje/catalogs/allmaps_compaso_cen_sat.npz` | every MAP matched to a halo at z_cat = 1.1 (121,386,086 rows) |
| `elg_catalog_base/step0_mass_window.npz` | the logM [11.3, 12.0) window, pre-applied (16,900,615 rows) |
| `elg_all_r95_AuroraHighBase_dje/catalogs/elgbase_sidecar_w11p3_12p0_{nall,sat}.npz` | six-epoch maplog quantities: `ram_int`, `dmult` |
| `elg_all_r95_AuroraHighBase_dje/catalogs/allmaps_compaso_primaries.npz` | the halo primaries — **the whole box**, 47,654,600 rows spanning logM 10.63–13.81, not just the window |
| `.../catalogs/lrg_widened/lrg_widened_lowdmult_n3p4e4.npz` | the base LRG |

Building the first three is **stage 0**: batch jobs on a compute node, hours,
not notebook cells. They already exist; these scripts start from them.
`elgmaps_core.ball_match` is the real matching algorithm written out so it can
be read, and `demo_ball_match()` runs it on random data.

Memory: `load_parent_window()` holds the 16.3M-row window plus the sidecar
joins — budget roughly **8–12 GB**. Pass `with_sidecar=False` for everything
except the selection rules themselves.

## Where the halos came from

| | |
|---|---|
| box | `AbacusSummit_highbase_c000_ph100`, snapshot `z1.100` — the **public** AbacusSummit HighBase box, 3456³ in 1000 Mpc/h, Mp = 2.109 × 10⁹ |
| cleaned | yes — `CompaSOHaloCatalog(cleaned=True, subsamples=False)`, cleaned `N == 0` rows dropped |
| halo radius | **`r95_L2com`** |
| method | `ball` matching, overlap resolved `most_massive` |
| counts | 48,043,269 halos, 47,654,600 primaries; 74,809,159 of 121,386,086 MAPs matched (61.6%) |

`r100_L2com` is only the matching script's argparse default — the run used
`r95_L2com`, which is what the run tag `elg_all_r95_*` records. Note that
`r_over_R` in the matched file is nonetheless divided by `r100_L2com`, chosen
deliberately so that `r/R` stays comparable across the r95 and r100 matches.

The MAPs come from a maplog re-run of the *same* initial conditions
(`HighBase_map_nosort`). Same phase `ph100`, same cosmology — that is what
makes the two joinable.

## Environment

The clustering needs **Corrfunc 2.5.3**. Check before starting:

```bash
python -c "import numpy, scipy, Corrfunc; print(Corrfunc.__version__)"
```

Expect `2.5.3`. The **stock** Corrfunc is required — the DESI fork
(`pycorr-desi`) has a different `DDrppi` signature and will fail here.

`elgmaps_core.py` depends only on numpy, scipy and Corrfunc; `scipy` and
`Corrfunc` are imported lazily, inside the functions that need them, so the
two self-contained scripts (`elg_best_l45.py`, `lrg_plusnonelg.py`) need
numpy alone.
