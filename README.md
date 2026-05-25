# FPL Model Performance Report

This README reviews the model's recorded performance from GW24 to GW38.

For implementation details, setup, commands, and model architecture, see [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Scope

The model was not run for the full season, so these results should be treated as a partial-season review rather than a complete assessment. GW23 has predictions recorded, but no actual or average score was recorded in the markdown file, so it is excluded from the scored evaluation.

Captaincy was not predicted by the model. Captains were selected manually and noted in the weekly markdown files using entries such as `14 (28 Cap)`. For model-only evaluation, the extra captain bonus is removed from the actual score.

The model-only actual score is calculated as:

```text
model-only actual = actual total - captain raw points
```

For example, if a captain scored `14 (28 Cap)`, the model-only score removes the extra `14` captain points but keeps the player's original `14` points as part of the selected XI.

## Overall Results

| Metric | Result |
| --- | ---: |
| Gameweeks evaluated | 15 |
| Evaluation window | GW24-GW38 |
| Predicted XI total | 812.58 |
| Model-only actual total | 820.00 |
| Difference | +7.42 |
| Mean predicted XI | 54.17 |
| Mean model-only actual | 54.67 |
| Bias | +0.49 pts/week |
| Mean absolute error | 8.91 |
| RMSE | 9.79 |
| Within 10 points | 9 / 15 |

The model was very well calibrated in aggregate. Across the evaluated period it predicted `812.58` starter points and the model-only actual total was `820.00`, a difference of only `7.42` points across 15 gameweeks.

This means the model was not meaningfully over- or under-predicting overall once manual captaincy is removed. The average bias was only `+0.49` points per week.

## Weekly Model-Only Accuracy

Positive error means the model underpredicted. Negative error means the model overpredicted.

| GW | Predicted XI | Actual Total | Captain Raw | Model-Only Actual | Error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 46.87 | 66 | 10 | 56 | +9.13 |
| 25 | 48.96 | 53 | 2 | 51 | +2.04 |
| 26 | 63.80 | 54 | 7 | 47 | -16.80 |
| 27 | 63.98 | 54 | 3 | 51 | -12.98 |
| 28 | 51.10 | 47 | 2 | 45 | -6.10 |
| 29 | 45.52 | 57 | 4 | 53 | +7.48 |
| 30 | 52.77 | 63 | 3 | 60 | +7.23 |
| 31 | 49.67 | 50 | 13 | 37 | -12.67 |
| 32 | 47.48 | 63 | 4 | 59 | +11.52 |
| 33 | 82.42 | 109 | 16 | 93 | +10.58 |
| 34 | 47.64 | 56 | 5 | 51 | +3.36 |
| 35 | 47.48 | 58 | 5 | 53 | +5.52 |
| 36 | 64.58 | 61 | 11 | 50 | -14.58 |
| 37 | 48.82 | 65 | 9 | 56 | +7.18 |
| 38 | 51.49 | 72 | 14 | 58 | +6.51 |

## Full Team Outcome

Although captaincy was manual, the full weekly team score is still useful for reviewing the combined process of model selection plus human captaincy.

| Metric | Result |
| --- | ---: |
| Full actual points | 928 |
| FPL average points | 770 |
| Difference vs average | +158 |
| Weeks beating average | 10 / 15 |
| Average weekly score | 61.87 |
| Average FPL average | 51.33 |

The overall team outcome was strong. The selected teams beat the FPL average by `158` points across the evaluated period and outperformed the average in `10` of `15` gameweeks.

## Strengths

### Strong Aggregate Calibration

Across GW24-GW38, the model predicted `812.58` starter points and the selected XI actually scored `820.00` points after removing the extra manual captain bonus. That is a difference of only `7.42` points across 15 gameweeks, or `0.49` points per week.

This is the clearest positive finding from the review: the model was very close in aggregate, even though individual gameweek errors were larger.

The earlier impression that the model was heavily underpredicting came from comparing non-captain-adjusted predicted starter points against actual totals that included manual captain doubling. Once that is corrected, the model is close to neutral in aggregate.

### Useful Team Selection Signal

The combined process produced teams that comfortably beat the FPL average. While this includes manual captaincy, the selected squads still needed to contain enough productive players for captaincy to matter.

The model appears useful as a selection and ranking tool, especially for building a playable XI under budget constraints.

### Good Performance in High-Scoring Weeks

The model contributed to several strong weeks, including GW33, GW38, GW30, GW34, and GW32. These weeks drove a large part of the overall outperformance versus the FPL average.

### Predictions Are Directionally Plausible

The prediction totals were usually in a realistic FPL range. The model was not producing obviously inflated or deflated weekly totals over the full evaluated period.

## Weaknesses

### Limited Evaluation Window

The model was only evaluated from GW24 to GW38, with GW23 excluded due to missing actuals. This is not enough to make a final judgement on season-long reliability.

A full-season evaluation could reveal issues that are hidden in this shorter sample, especially around early-season uncertainty, fixture swings, rotation, injuries, and promoted-team effects.

### Week-to-Week Volatility

The aggregate result is strong, but individual gameweek errors are still material.

The model-only MAE was `8.91` points and RMSE was `9.79` points. It landed within 10 points in `9` of `15` weeks, but there were still several larger misses.

Largest overpredictions:

| GW | Predicted XI | Model-Only Actual | Error |
| ---: | ---: | ---: | ---: |
| 26 | 63.80 | 47 | -16.80 |
| 36 | 64.58 | 50 | -14.58 |
| 27 | 63.98 | 51 | -12.98 |
| 31 | 49.67 | 37 | -12.67 |

Largest underpredictions:

| GW | Predicted XI | Model-Only Actual | Error |
| ---: | ---: | ---: | ---: |
| 32 | 47.48 | 59 | +11.52 |
| 33 | 82.42 | 93 | +10.58 |
| 24 | 46.87 | 56 | +9.13 |

This suggests the model is better judged over multiple weeks than as an exact weekly score forecast.

### No Modelled Captaincy

Captaincy was selected manually, not by the model. This is fine operationally, but it means full team score cannot be attributed entirely to the model.

For future reviews, model selection and captain selection should continue to be evaluated separately.

### Limited Insight Into Player-Level Accuracy

This review focuses on weekly team totals. It does not yet answer whether the model is consistently ranking individual players correctly within positions.

Useful future checks would include:

- Top-ranked players vs lower-ranked alternatives by position
- Predicted points buckets vs actual returns
- How often the model's selected starters beat the bench alternatives
- Whether high predicted scores are actually producing high returns

### FPL Scoring Is High Variance

FPL points are driven by discrete events such as goals, assists, clean sheets, bonus points, cards, substitutions, and injuries. Even a well-calibrated expected-points model will miss individual weeks.

The model should therefore be treated as a decision-support tool, not a precise score predictor.

## Interpretation

The model performed well in the period evaluated.

The most important finding is that, after removing the manual captain bonus, the model's total prediction was almost exactly aligned with actual outcomes. This indicates good aggregate calibration.

The model also supported a team-selection process that beat the FPL average by a wide margin. That is the practical objective, and the process achieved it over the recorded period.

The main limitation is sample size. Fifteen scored gameweeks is useful, but not enough to claim that the model is proven over a full season. The review also does not yet isolate player-level ranking quality or compare selected players against realistic alternatives not chosen by the optimizer.

## Conclusion

The model looks promising and useful.

It was almost perfectly calibrated in aggregate across GW24-GW38, with only `7.42` points of total difference between predicted XI points and model-only actual points. The full team process also beat the FPL average by `158` points across 15 gameweeks.

The strongest conclusion is that the model is a good selection aid over a multi-week period. The weaker conclusion is that it can accurately forecast any single gameweek, because weekly variance remains significant and the evaluation sample is still limited.
