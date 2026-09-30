# Data Dictionary

The processed dataset contains one record per FBS-vs-FBS game. The exact rolling feature list is generated from the available CFBD fields, but all features follow the definitions below.

## Identifiers and Context

| Feature | Description |
| --- | --- |
| `game_id` | CollegeFootballData game identifier |
| `season` | College football season |
| `week` | Week within the season |
| `start_date` | Scheduled kickoff time in UTC |
| `home_team`, `away_team` | Participating teams |
| `neutral_site` | 1 for a neutral-site game and 0 otherwise |
| `conference_game` | 1 when CFBD identifies a conference game |
| `postseason` | 1 for a postseason game |
| `covid_season` | Indicator for the unusual 2020 season |

## Rolling Features

The prefix identifies the team (`home_` or `away_`). The suffix identifies the historical summary:

| Suffix | Description |
| --- | --- |
| `_last_5` | Mean over the team's five most recent completed games |
| `_ewm` | Exponentially weighted mean that emphasizes recent games |
| `_season` | Mean over completed games in the current season |

Examples include points scored and allowed, yards, plays, turnovers, EPA, success rate, explosiveness, third-down performance, and finishing drives.

Features beginning with `diff_` equal the home value minus the away value.

## Ratings and Program Strength

| Feature | Description |
| --- | --- |
| `home_elo`, `away_elo` | Pregame Elo ratings calculated within this project |
| `elo_diff` | Home Elo minus away Elo |
| `talent_score` | CFBD team talent composite |
| `recruiting_points` | CFBD team recruiting score for the class year |

## Targets

| Feature | Description |
| --- | --- |
| `actual_margin` | Home points minus away points |
| `actual_total` | Home points plus away points |
| `home_points`, `away_points` | Observed final scores, including overtime |

## Market Fields

| Feature | Description |
| --- | --- |
| `market_margin` | Consensus market prediction of the home-team margin |
| `market_total` | Median over/under across available providers |
| `line_providers` | Number of providers used in the consensus |

The market fields are excluded from the statistics-only model.

## Prediction Fields

| Feature pattern | Description |
| --- | --- |
| `model_<family>_margin` | Margin prediction from one base model or the stacking ensemble |
| `model_<family>_total` | Total-points prediction from one base model or the stacking ensemble |
| `model_margin`, `model_total` | Predictions from the systems selected on the 2024 season |
| `model_home_score`, `model_away_score` | Scores reconstructed from the selected margin and total predictions |
| `model_margin_low`, `model_margin_high` | Empirical 90% margin interval calibrated on 2024 absolute errors |
| `model_total_low`, `model_total_high` | Empirical 90% total interval calibrated on 2024 absolute errors |
| `hybrid_*` | Separate statistics-plus-market predictions when lines are available |
