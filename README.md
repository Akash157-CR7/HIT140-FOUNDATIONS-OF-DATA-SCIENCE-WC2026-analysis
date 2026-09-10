# FIFA World Cup 2026 — Objective 1

**Darwin Group 46**  ·  Assessment 2, due 10 September 2026, 14:00 ACST

| Member | Student ID | Task owned |
|---|---|---|
| Angel Shahi | s400420 | 1 - Goal timing |
| Nishant Shrestha | s400342 | 2 - Team playing style |
| Sabraham Shrestha | s404389 | 3 - Attacking output by position |
| Akash Singh | s401643 | 4 -Schedule recovery |

---

## The four analytic tasks

| # | Owner | Focal point | Response variable | Unit of analysis | n | Test |
|---|---|---|---|---|---|---|
| 1 | Angel Shahi | Goal timing - **when** a goal arrives | Minute of the goal | Goal event | 300 | **One-sample** *t* against 45.5 |
| 2 | Nishant Shrestha | Team playing style - **how** a team plays | Possession share (%) | Nation | 48 | Two-sample *t* |
| 3 | Sabraham Shrestha | Attacking output - **who** creates goals | Goals + assists per 90 min | Player | 286 | Two-sample *t* |
| 4 | Akash Singh | Schedule recovery - **how often** they play | Days between matches | Recovery interval | 160 | Two-sample *t* |

Four different constructs, four different response variables, four different units of
analysis. Every task covers the six required skills: analytic question, data wrangling,
data preparation and sampling, descriptive statistics, confidence interval, and a *t*-test.

### Results

| # | Mean | 95% CI (z-interval) | *t* | df | p | Decision |
|---|---|---|---|---|---|---|
| 1 | 52.65 min | [49.58, 55.73] | 4.56 | 299 | < .001 | Reject H₀ |
| 2 | 48.63 % | [45.95, 51.30] | 2.63 | 15 | .009 | Reject H₀ |
| 3 | 0.34 per 90 | [0.30, 0.39] | 4.42 | 72 | < .001 | Reject H₀ |
| 4 | 5.29 days | [5.14, 5.44] | −4.12 | 63 | < .001 | Reject H₀ |

The interval in each row is the Week 3 z-interval for that task's own mean -
x̄ ± z*·s/√n with z* = 1.960 - not for the difference between groups.

Each task also reports a second result that qualifies the headline:

| # | The check | What it shows |
|---|---|---|
| 1 | Re-run without added-time goals | t(253) = 1.56, p = .120 - the late surplus sits in added time |
| 2 | Split by confederation instead | t(21) = 2.93, p = .004 - style and strength are tangled |
| 3 | Test assists alone | t(72) = 1.46, p = .074 - the choice of variable carries the task |
| 4 | Paired winner-vs-loser rest | t(31) = −0.39, p = .701 - the squeeze is even-handed |

The four focal points were chosen after exploring the verified data, so running four
tests raises the chance that at least one rejection is a false positive. HIT140 has
not covered a correction for that, so the fact is **stated rather than silently
corrected or silently ignored**. Every one of the four is decided well below α = 0.05.

## Method scope

The brief requires the skills of **Weeks 1 to 5**. Every procedure used here comes from that range and nothing beyond it:

| Week | What is used |
|---|---|
| 2 | mean, median, range, quartiles, IQR, the **1.5 × IQR outlier rule**, variance, standard deviation, skewness, histogram |
| 3 | Central Limit Theorem, z* = 1.960, **CI = x̄ ± z*·s/√n** with its n ≥ 30 condition |
| 4 | the **four-step process** (State, Plan, Solve, Conclude), the t-distribution, df = n − 1, the one-sample *t*, the two-sample independent *t* with separate variances and the conservative **df = min(n₁,n₂) − 1**, α = 0.05, APA reporting |
| 5 | pandas for loading, filtering and grouping |

Techniques outside that range are deliberately absent: bootstrap intervals,
Levene's test, Welch degrees of freedom, pooled variance, Mann–Whitney, Wilcoxon,
Shapiro–Wilk, Cohen's *d*, statistical power, the Bonferroni correction - and
correlation, which is Week 6.

Both *t*-tests are written out from the Week 4 formulas rather than called from a
library, so the arithmetic in the code is the arithmetic in the lecture.

---

## What is in this folder

| Path | What it is |
|---|---|
| `WC2026_Analysis.ipynb` | **The analysis.** All four tasks, figures inline, outputs saved with the file. |
| `WC2026_Analysis.py` | The same code as a plain script, for reading or running outside Jupyter. |
| `dataset/*.csv` | The seven data tables the notebook reads. |
| `requirements.txt` | The four packages the analysis needs. |
| `README.md` | This file. |

### The dataset is exactly what the four tasks read

| Table | Rows | Serves |
|---|---|---|
| `goals.csv` | 308 | Task 1 - minute of each goal event |
| `squad_profile.csv` | 48 | Task 2 - possession share per nation |
| `players.csv` | 1,039 | Task 3 - goal involvements per 90 |
| `rest.csv` | 160 | Task 4 - the recovery intervals |
| `rest_pairs.csv` | 32 | Task 4 - the paired winner-vs-loser test |
| `matches.csv` | 104 | verification - match count and stages |
| `team_match.csv` | 208 | verification - team-match count |

### The variables were built, not found

None of the four response variables existed in a source file:

| Task | Variable | Built from |
|---|---|---|
| 1 | `minute` | the published minute string, e.g. `90+4` |
| 2 | `possession_pct` | FBref squad possession table |
| 3 | `involvements_per90` | (goals + assists) / minutes × 90 |
| 4 | `rest_days` | days to the **next** match, per nation |

`rest_days` is the one worth reading twice: each interval is labelled by the match it
leads *into*, not the one it follows. Labelling it the obvious way hides the effect,
because the last group-stage interval is really the run-up to a Round of 32 tie.

## Data sources

**FBref is our primary source.** Its 2026 World Cup tables carry Tasks 2, 3 and 4
outright - squad possession, the player tables, and the fixture list with dates.
They were taken from the site's own *Share & Export → Get table as CSV* control.

FBref publishes goal times only inside individual match reports, not in any
exportable table, so the 308 goal events with their minutes come from an open
archive. **Task 1 alone depends on it.** A third archive was used for verification
only — no task reads it.

| | Source | Role | Used by |
|---|---|---|---|
| 1 | **FBref, 2026 World Cup** (`fbref.com`) | primary - players, squads, fixtures | Tasks 2, 3, 4 |
| 2 | **FIFA 2026 (`fifa2026.com`) | the 308 goal events and their minutes | Task 1 |


Three sources maintained by different people from different upstreams agree on all
104 scorelines and all 308 goal events. That agreement is only possible if all three
describe the same tournament, so every FBref figure is corroborated before it is
analysed.

## Running it

Python 3.10 or newer. Everything runs from inside this folder.

### The notebook (the primary artefact)

`WC2026_Analysis.ipynb` walks through all four tasks with **the figures inline** and
every table printed. It is saved with its outputs already in it, so it can be read
without running anything - open it in VS Code (install the *Jupyter* extension) or
with `jupyter notebook`. To re-run it, use **Run All**.

The first cell installs any missing package and locates `dataset/`, so **Run All**
works on a clean machine with nothing installed.

### The script

```bash
pip install -r requirements.txt   # optional; the first cell does this too
python3 WC2026_Analysis.py
```

On Windows use `py` instead of `python3`. The script prints the same numbers as the
notebook and opens each figure in a window.

### Google Colab

Open [colab.research.google.com](https://colab.research.google.com) → **Upload** →
choose `WC2026_Analysis.ipynb`, then run the first cell: it asks you to upload a zip
of this folder, unpacks it, and finds `dataset/`. Then *Runtime → Run all*.

The Colab branch of that cell does nothing when the notebook is not on Colab, so the
same file works in both places.

### Verification before analysis

The notebook will not proceed past its check cells unless the data is intact. Eight
row-count checks run first:

```
nations 48 · matches 104 · goal events 308 · normal-time goals 300
extra-time goals 8 · players 1039 · recovery intervals 160 · team-matches 208
```

Two stronger cross-file checks follow, each counting the same quantity from a
different table — goal events against player goals + own goals (308 = 308), and the
48 nations appearing in `rest.csv`. An `assert` stops the run if any check fails,
because analysing data that has drifted is worse than not analysing it.

### What a good run looks like

```
All 8 checks passed.
Both cross-checks passed.
...
  t(299) = 4.56, p < .001     REJECT H0
  t(15)  = 2.63, p = .009     REJECT H0
  t(72)  = 4.42, p < .001     REJECT H0
  t(63)  = -4.12, p < .001    REJECT H0
```

## What each task may and may not claim

| # | What we found | What we cannot say |
|---|---|---|
| 1 | Goals cluster late, mostly in added time | Remove added-time goals and the effect is no longer significant |
| 2 | Nations that advanced held more possession | Style and strength are tangled; this design cannot separate them |
| 3 | Forwards out-produce midfielders per 90 | Players are clustered within nations |
| 4 | The knockout stage shortens recovery | It does so even-handedly; winners get no advantage |

Task 2 carries one further limitation: the group-exit sample is 16 nations, below the
30 the Central Limit Theorem asks for, and it is right-skewed. The test is reported
anyway, with that stated rather than hidden.
