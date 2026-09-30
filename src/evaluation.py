import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(predictions, prefix="model"):
    """Calculate score, margin, total, and winner metrics."""
    completed = predictions.dropna(
        subset=["home_points", "away_points", f"{prefix}_margin", f"{prefix}_total"]
    )
    actual_winner = completed["actual_margin"].gt(0)
    predicted_winner = completed[f"{prefix}_margin"].gt(0)
    return pd.Series(
        {
            "games": len(completed),
            "margin_mae": mean_absolute_error(
                completed["actual_margin"], completed[f"{prefix}_margin"]
            ),
            "margin_rmse": mean_squared_error(
                completed["actual_margin"], completed[f"{prefix}_margin"]
            ) ** 0.5,
            "total_mae": mean_absolute_error(
                completed["actual_total"], completed[f"{prefix}_total"]
            ),
            "total_rmse": mean_squared_error(
                completed["actual_total"], completed[f"{prefix}_total"]
            ) ** 0.5,
            "home_score_mae": mean_absolute_error(
                completed["home_points"], completed[f"{prefix}_home_score"]
            ),
            "away_score_mae": mean_absolute_error(
                completed["away_points"], completed[f"{prefix}_away_score"]
            ),
            "winner_accuracy": accuracy_score(actual_winner, predicted_winner),
            "margin_r2": r2_score(completed["actual_margin"], completed[f"{prefix}_margin"]),
            "total_r2": r2_score(completed["actual_total"], completed[f"{prefix}_total"]),
        }
    )


def add_market_score_predictions(data):
    output = data.copy()
    output["market_home_score"] = np.maximum(
        0, (output["market_total"] + output["market_margin"]) / 2
    )
    output["market_away_score"] = np.maximum(
        0, (output["market_total"] - output["market_margin"]) / 2
    )
    return output


def add_baseline_predictions(test_data, training_data):
    """Add a historical-average baseline and a recent scoring baseline."""
    output = test_data.copy()

    output["naive_margin"] = training_data["actual_margin"].mean()
    output["naive_total"] = training_data["actual_total"].mean()
    output["naive_home_score"] = (output["naive_total"] + output["naive_margin"]) / 2
    output["naive_away_score"] = (output["naive_total"] - output["naive_margin"]) / 2

    output["rolling_home_score"] = output[
        ["home_points_for_ewm", "away_points_against_ewm"]
    ].mean(axis=1)
    output["rolling_away_score"] = output[
        ["away_points_for_ewm", "home_points_against_ewm"]
    ].mean(axis=1)
    output["rolling_margin"] = output["rolling_home_score"] - output["rolling_away_score"]
    output["rolling_total"] = output["rolling_home_score"] + output["rolling_away_score"]
    return output


def paired_bootstrap_mae_difference(
    predictions, actual_column, first_prediction, second_prediction, n_bootstrap=2000, random_state=42
):
    """Bootstrap paired games; negative values favor the first prediction."""
    data = predictions[[actual_column, first_prediction, second_prediction]].dropna()
    first_error = np.abs(data[actual_column] - data[first_prediction]).to_numpy()
    second_error = np.abs(data[actual_column] - data[second_prediction]).to_numpy()
    difference = first_error - second_error
    rng = np.random.default_rng(random_state)
    bootstrap = np.empty(n_bootstrap)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(difference), len(difference))
        bootstrap[index] = difference[sample].mean()
    return {
        "mean_mae_difference": float(difference.mean()),
        "lower_95": float(np.quantile(bootstrap, 0.025)),
        "upper_95": float(np.quantile(bootstrap, 0.975)),
    }


def prediction_interval_coverage(predictions, actual, lower, upper):
    data = predictions[[actual, lower, upper]].dropna()
    covered = data[actual].between(data[lower], data[upper])
    return pd.Series(
        {
            "coverage": covered.mean(),
            "average_width": (data[upper] - data[lower]).mean(),
            "games": len(data),
        }
    )
