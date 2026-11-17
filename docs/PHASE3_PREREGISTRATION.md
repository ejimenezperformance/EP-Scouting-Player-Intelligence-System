# Phase 3 — pre-registered design (hitters MVP)

Written **before** any model was fit on real data. Changes after this point must be logged in the README corrections log.

## Decision question
At the end of season *t*, which MLB hitters will produce **above** (breakout) or **below** (regression) what a simple projection expects in season *t+1*?

## Population and labels
- Eligible at *t*: ≥ 200 PA in season *t* (≥ 80 PA in 2020, a 60-game season).
- Target: wOBA in *t+1* for hitters with ≥ 200 PA in *t+1* (≥ 80 in 2020).
- Players who fall below that PA in *t+1* (or disappear) are **not dropped**: they get the label `lost_role = 1` and are reported separately.
- Residual = actual wOBA(*t+1*) − Marcel wOBA(*t+1*).
- `breakout` = residual ≥ +0.020; `regression` = residual ≤ −0.020.

## Baseline: Marcel-style projection
- wOBA from seasons *t*, *t−1*, *t−2*, weighted 5/4/3 by PA, regressed toward the league average with a constant *K* fit **on training seasons only**, plus the Marcel age adjustment (+0.006 per year under 29, −0.003 per year over 29, as a multiplier).

## Amendment 1 (before any model was fit)
Marcel's residuals were negative on average in the training seasons (league offense fell in several years).
To make the baseline harder to beat, the comparison baseline is **Marcel + a constant offset equal to the mean
training residual**. `breakout` and `regression` labels use this bias-corrected Marcel. *K* was fit on train
target seasons over a grid of 1,000–10,000 weighted PA (chosen: 3,000).

## Features (season *t* and earlier only — no future data)
wOBA, xwOBA, xwOBA − wOBA, xSLG, xBA, Barrel%, Hard-Hit%, average EV, launch angle, Sweet-Spot%, K%, BB%, Whiff%, Chase%, sprint speed, age, PA, one-year changes in xwOBA, Barrel%, EV, K%, and the Marcel projection itself.
Bat tracking (bat speed, attack angle) is **excluded** from the MVP: it only exists from 2023–2024, too few seasons to train and test.

## Chronological split
| Set | Target seasons (*t+1*) | Use |
|---|---|---|
| Train | 2017–2022 | Fit models |
| Validation | 2023–2024 | Choose hyperparameters / model |
| Test | 2025–2026 | Touched once, at the end |

## Models
1. Marcel (baseline)
2. Ridge regression on the residual
3. XGBoost on the residual
4. Logistic regression and XGBoost classifiers for `breakout` and `regression`

## Success criteria (test set, 2025–2026)
- **Regression models — `CONFIRMED` adds value over Marcel** if the MAE improvement over Marcel is ≥ 0.002 wOBA **and** its 95% paired cluster bootstrap CI (resampling players) excludes 0.
- **Classifiers — `CONFIRMED` useful** if AUC 95% CI lower bound > 0.55 **and** Brier skill score vs the base rate > 0.
- Anything else is reported as `NOT CONFIRMED`, with the same prominence.
- Probabilities are shown on dashboards only for models that pass.

## Known biases handled
- **Survivorship:** `lost_role` kept and analyzed.
- **Leakage:** every feature is from season *t* or earlier; Marcel's *K* fit on train only; test set used once.
- **Short season:** 2020 PA thresholds scaled.
- **Context not modeled in MVP:** park, position, team changes, injuries — listed as limitations.
