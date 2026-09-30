# Predicting FBS College Football Scores with Machine Learning

**Author:** Alexander Parks

This project predicts the final score of FBS college football games using information that is available before kickoff. Six statistical and machine learning models are tuned with chronological validation, and an out-of-fold stacking ensemble tests whether combining them improves performance. Betting information is excluded from this process. The selected statistics-only system is then compared to consensus betting lines, and a separate hybrid tests whether market information improves the results.

The project was designed as a traditional data science study. It includes data collection, cleaning, exploratory analysis, feature engineering, model selection, chronological validation, final evaluation, and a notebook that can be rerun each week.

## Main Questions

1. How accurately can final scores be predicted from prior team performance, advanced efficiency statistics, Elo ratings, and program talent?
2. Which model performs best on future seasons rather than randomly selected games?
3. How does the statistics-only model compare to the betting market?
4. Does a hybrid of the model and market improve either prediction alone?

## Prediction Method

The models predict two values:

- `margin = home score - away score`
- `total = home score + away score`

These are converted back to team scores:

- `home score = (total + margin) / 2`
- `away score = (total - margin) / 2`

Margin and total are modeled separately because they represent different parts of a game. Margin primarily measures the difference in team strength, while total reflects the expected scoring environment.

## Repository Structure

```text
college-football-score-prediction/
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_cleaning_and_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_and_tuning.ipynb
│   ├── 05_final_evaluation.ipynb
│   └── 06_weekly_predictions.ipynb
├── src/
├── data/
├── docs/
├── models/
└── outputs/
```

The notebooks should be run in order the first time. After the historical model has been created, only the sixth notebook is needed for a normal weekly update.

## Setup

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

### 2. Obtain a free CFBD API key

1. Register at [CollegeFootballData](https://collegefootballdata.com/).
2. Use the free Academic tier if a `.edu` email address is available.
3. Create a file named `.env` in the project root.
4. Add `CFBD_API_KEY=your_key` using the key you received.

The `.env` file is ignored by Git and should never be committed.

### 3. Open the notebooks

```bash
jupyter lab
```

The project uses only free data and open-source Python packages. A GPU is optional. The PyTorch model automatically uses CUDA when it is available and otherwise runs on CPU.

### Google Colab

1. Upload the repository to Google Drive or clone it into Colab.
2. Select **Runtime > Change runtime type > GPU** if a free GPU is available.
3. Run `pip install -r requirements.txt` in a Colab cell.
4. Create `.env` in the project root and add `CFBD_API_KEY=your_key`.
5. Run the notebooks in order.

The full model search is the default in Notebook 04. On a CPU it may take substantially longer than the other notebooks, but it is designed to finish within a normal Colab session. `FAST_MODE = True` is available only for a short code check and should not be used for the reported analysis.

## Data

The primary source is the [CollegeFootballData API](https://api.collegefootballdata.com/getting-started). The model uses FBS-vs-FBS games beginning in 2014. Historical API responses are cached locally so they do not need to be downloaded again.

In accordance with CollegeFootballData's API terms, raw, normalized, and row-level source datasets are not redistributed in this repository. After a user creates `.env` with their own API key, Notebooks 01 through 03 recreate the excluded data locally. Compact derived metrics, figures, and executed notebook results remain included for review.

The main feature groups are:

- Recent and season-to-date scoring
- Offensive and defensive yardage
- EPA, success rate, and explosiveness
- Third- and fourth-down performance
- Pace, possession, and turnovers
- Pregame Elo
- Recruiting and roster talent
- Rest and game context

Betting lines are stored in the final modeling table but explicitly excluded from the statistics-only model features.

## Validation Design

College football changes over time, so a random split would allow the model to learn from games that occurred after games in the validation set. Instead, this project uses expanding-season validation. Hyperparameters are tuned on future-season folds from 2020 through 2023. The 2024 season is then used to select between each individual model and the ensemble and to calibrate prediction intervals. The 2025 season is evaluated once as the untouched final test.

For weekly use, all six base models are retrained on the latest completed games with their fixed hyperparameters. The selected system and ensemble weights are then applied to the upcoming schedule.

## Models

The project compares:

- Simple score and Elo baselines
- Ridge regression
- Elastic Net
- Random Forest
- Histogram Gradient Boosting
- RBF Support Vector Regression
- A multi-layer PyTorch neural network
- A nonnegative linear stacking ensemble

The models provide a useful progression from regularized linear relationships to bagging, boosting, kernel methods, and deep learning. The neural network uses ReLU hidden layers, dropout, AdamW, weight decay, Huber loss, and chronological early stopping. The ensemble is fitted only on out-of-fold base-model predictions. Logistic regression and Naive Bayes are not forced into the comparison because the primary targets are continuous rather than categorical.

## Results

Gradient Boosting produced the lowest tuning MAE for margin, while Elastic Net produced the lowest tuning MAE for total points. On the later 2024 selection season, Random Forest was selected for margin and the stacking ensemble was selected for total. The final models were trained on 8,277 games from 2014-2024 and evaluated once on 808 games from the 2025 season.

| Metric | Statistics-only model | Market | Hybrid |
| --- | ---: | ---: | ---: |
| Margin MAE | 12.55 | 11.82 | 11.82 |
| Total MAE | 12.65 | 12.54 | 12.62 |
| Home score MAE | 9.40 | 9.05 | 9.15 |
| Away score MAE | 8.51 | 8.14 | 8.15 |
| Winner accuracy | 72.9% | 73.6% | 73.6% |

The market's margin MAE was 0.72 points lower than the statistics-only model. A paired bootstrap 95% interval of 0.37 to 1.06 points suggests that this was a stable difference. The market's total MAE was only 0.11 points lower, with an interval that crossed zero. The simple hybrid model did not improve on the market.

These results are useful even though the statistics-only model did not beat the market. They show that the model learned a meaningful signal from public football data, while the comparison provides an honest standard for judging its practical performance.

## Important Limitations

- College football teams change substantially between seasons.
- Injuries and late roster news are not modeled consistently.
- Early-season predictions rely heavily on prior information.
- Historical lines may not represent the exact line available at one consistent time before kickoff.
- A good prediction model is not automatically a profitable betting model.

See `docs/model_card.md` for a more complete description of intended use and limitations. The complete interpretation is available in `docs/final_report.md`.
