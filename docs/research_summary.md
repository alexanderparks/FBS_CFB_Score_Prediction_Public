# Research Summary

## Problem

The purpose of this project is to predict final scores for FBS college football games. A useful model must do more than fit completed games. It must reproduce the information restrictions that exist when a future game is predicted.

## Prior Approaches

Prior college football models commonly use team ratings, rolling box-score statistics, recruiting information, and advanced efficiency statistics. South and Egros compared regularized regression, nearest neighbors, neural networks, random forests, gradient boosting, and a Bayesian model. They trained on the 2011-2014 seasons and evaluated on the 2015 season, which supports the use of a future-season test rather than a random split.

Efficiency-based football analysis commonly focuses on expected points added, success rate, explosiveness, finishing drives, field position, and opponent strength. These measures help distinguish sustainable performance from final scores that may have been influenced by turnovers or a small number of unusual plays.

## Decisions for This Project

1. The dataset includes only games where both teams were members of the FBS.
2. The model predicts margin and total, then reconstructs the two final scores.
3. Every rolling feature is shifted in time so the target game is not included.
4. Model selection uses expanding-season validation.
5. The 2025 season is held out for final evaluation.
6. Betting lines are not available to the original model.
7. The market is treated as a separate benchmark before being considered as a hybrid feature.
8. Six model families represent regularized regression, bagging, boosting, kernel learning, and deep learning.
9. A linear stacking model is fitted on out-of-fold predictions so it never learns from in-sample base-model predictions.
10. The 2024 season is reserved for model and ensemble selection before the untouched 2025 test.

## Sources

- CollegeFootballData. [Getting Started](https://api.collegefootballdata.com/getting-started).
- CollegeFootballData. [API Reference](https://apinext.collegefootballdata.com/api/games).
- SportsDataverse. [cfbfastR](https://github.com/sportsdataverse/cfbfastR).
- South, C. and Egros, E. (2020). [Forecasting college football game outcomes using modern modeling techniques](https://journals.sagepub.com/doi/10.3233/JSA-190314).
