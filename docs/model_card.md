# Model Card

## Model Purpose

The model produces pregame final-score predictions for games between two FBS teams. It is intended for data science analysis and weekly personal forecasting.

## Inputs

The statistics-only model uses prior game results, rolling team statistics, advanced efficiency, Elo, recruiting, talent, rest, and game context. It does not use a point spread, over/under, or moneyline.

## Outputs

- Predicted home and away score
- Predicted home margin
- Predicted game total
- Prediction intervals for margin and total
- A separate market prediction when a line is available

## Evaluation

Six model families are tuned with expanding-season validation through 2023. A stacking model learns nonnegative weights from out-of-fold predictions. The 2024 season is used to select between the individual models and the ensemble and to calibrate prediction intervals. The 2025 season is reserved for final evaluation.

The selected margin model was Random Forest, and the selected total model was the stacking ensemble. On 808 games from the 2025 test season, the statistics-only system obtained:

- Margin MAE: 12.55 points
- Total MAE: 12.65 points
- Home score MAE: 9.40 points
- Away score MAE: 8.51 points
- Winner accuracy: 72.9%

The nominal 90% intervals covered 89.4% of observed margins and 89.2% of observed totals.

## Intended Use

- Demonstrating an end-to-end data science workflow
- Comparing regularized regression, tree ensembles, a nonlinear kernel, deep learning, and stacking on time-ordered sports data
- Producing a new set of predictions once per week
- Studying the difference between a statistical model and the betting market

## Uses Outside the Project's Scope

- Live, in-game prediction
- Player proposition prediction
- Automatic wagering
- Claims of guaranteed financial return

## Limitations

The model does not consistently account for injuries, late depth-chart changes, weather, or changes in player availability. Early-season samples are small, and roster turnover makes information from prior seasons less reliable. Betting data may also represent different collection times across providers.

The market was a stronger margin predictor on the 2025 test season. The market and statistics-only model performed similarly on total points. A linear hybrid did not improve on the market, so it is retained as an experiment rather than presented as the final model.
