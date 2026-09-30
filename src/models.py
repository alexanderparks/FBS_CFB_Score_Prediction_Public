import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import RANDOM_STATE
from src.deep_learning import TorchMLPRegressor


class StackedTargetPredictor(RegressorMixin, BaseEstimator):
    """Prediction-only wrapper used to interpret a fitted stacking ensemble."""

    def __init__(self, models, stacker, model_names):
        self.models = models
        self.stacker = stacker
        self.model_names = model_names

    def predict(self, X):
        base_predictions = pd.DataFrame(
            {name: self.models[name].predict(X) for name in self.model_names}
        )
        return self.stacker.predict(base_predictions[self.model_names])


def candidate_models(fast_mode=False):
    """Return the six model families and bounded parameter grids used in the comparison."""
    scaled_steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    models = {
        "Ridge": (
            Pipeline(scaled_steps + [("model", Ridge())]),
            {"model__alpha": [1.0, 10.0] if fast_mode else [0.1, 1.0, 10.0, 100.0]},
        ),
        "Elastic Net": (
            Pipeline(
                scaled_steps
                + [("model", ElasticNet(max_iter=10000, random_state=RANDOM_STATE))]
            ),
            {
                "model__alpha": [0.01, 0.1] if fast_mode else [0.001, 0.01, 0.1, 1.0],
                "model__l1_ratio": [0.25] if fast_mode else [0.2, 0.5, 0.8],
            },
        ),
        "Random Forest": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=150 if fast_mode else 400,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [10] if fast_mode else [8, 14],
                "model__min_samples_leaf": [4] if fast_mode else [2, 5],
                "model__max_features": [0.7] if fast_mode else [0.5, 0.8],
            },
        ),
        "Gradient Boosting": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            early_stopping=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__learning_rate": [0.06] if fast_mode else [0.03, 0.07],
                "model__max_leaf_nodes": [31] if fast_mode else [15, 31],
                "model__l2_regularization": [1.0] if fast_mode else [1.0, 5.0],
            },
        ),
        "RBF Support Vector Regression": (
            Pipeline(scaled_steps + [("model", SVR(kernel="rbf"))]),
            {
                "model__C": [10.0] if fast_mode else [1.0, 10.0, 50.0],
                "model__epsilon": [0.5] if fast_mode else [0.1, 1.0],
                "model__gamma": ["scale"] if fast_mode else ["scale", 0.01],
            },
        ),
        "Neural Network": (
            Pipeline(
                scaled_steps
                + [
                    (
                        "model",
                        TorchMLPRegressor(
                            batch_size=1024,
                            max_epochs=30 if fast_mode else 160,
                            patience=5 if fast_mode else 15,
                            random_state=RANDOM_STATE,
                            device="auto",
                        ),
                    )
                ]
            ),
            {
                "model__hidden_layer_sizes": [(128, 64)]
                if fast_mode
                else [(128, 64), (256, 128, 64)],
                "model__dropout": [0.20] if fast_mode else [0.10, 0.30],
                "model__learning_rate": [0.001]
                if fast_mode
                else [0.001, 0.0003],
                "model__weight_decay": [0.0001],
            },
        ),
    }
    return models


def expanding_season_folds(data, first_validation_season=None, last_validation_season=None):
    seasons = sorted(int(season) for season in data["season"].dropna().unique())
    first_validation_season = first_validation_season or seasons[max(3, len(seasons) - 4)]
    last_validation_season = last_validation_season or seasons[-1]
    folds = []
    for validation_season in seasons:
        if first_validation_season <= validation_season <= last_validation_season:
            train_index = data.index[data["season"] < validation_season].to_numpy()
            validation_index = data.index[data["season"] == validation_season].to_numpy()
            if len(train_index) and len(validation_index):
                folds.append((train_index, validation_index, validation_season))
    return folds


def tune_models(data, features, target, folds, fast_mode=False, verbose=True):
    """Compare candidate settings using mean MAE across future-season folds."""
    rows = []
    best_estimators = {}
    for model_name, (pipeline, parameter_grid) in candidate_models(fast_mode).items():
        model_start = time.perf_counter()
        best_score = np.inf
        best_estimator = None
        settings = list(ParameterGrid(parameter_grid))
        if verbose:
            print(f"Tuning {model_name}: {len(settings)} parameter settings")
        for parameters in settings:
            fold_scores = []
            for train_index, validation_index, validation_season in folds:
                estimator = clone(pipeline).set_params(**parameters)
                estimator.fit(data.loc[train_index, features], data.loc[train_index, target])
                prediction = estimator.predict(data.loc[validation_index, features])
                fold_scores.append(mean_absolute_error(data.loc[validation_index, target], prediction))
            mean_score = float(np.mean(fold_scores))
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "parameters": parameters,
                    "mean_validation_mae": mean_score,
                    "fold_mae": fold_scores,
                }
            )
            if mean_score < best_score:
                best_score = mean_score
                best_estimator = clone(pipeline).set_params(**parameters)
        best_estimators[model_name] = best_estimator
        if verbose:
            elapsed_minutes = (time.perf_counter() - model_start) / 60
            print(f"  best validation MAE: {best_score:.3f} ({elapsed_minutes:.1f} minutes)")
    results = pd.DataFrame(rows).sort_values("mean_validation_mae").reset_index(drop=True)
    return results, best_estimators


def generate_oof_predictions(data, features, target, folds, estimators, verbose=True):
    """Create expanding-window out-of-fold predictions for a set of tuned estimators."""
    prediction_frames = []
    for fold_number, (train_index, validation_index, validation_season) in enumerate(folds, start=1):
        fold_predictions = pd.DataFrame(
            {
                "row_index": validation_index,
                "season": validation_season,
                "actual": data.loc[validation_index, target].to_numpy(),
            }
        )
        if verbose:
            print(f"OOF fold {fold_number}: predicting {validation_season}")
        for model_name, estimator in estimators.items():
            fitted_model = clone(estimator).fit(
                data.loc[train_index, features], data.loc[train_index, target]
            )
            fold_predictions[model_name] = fitted_model.predict(
                data.loc[validation_index, features]
            )
        prediction_frames.append(fold_predictions)
    return pd.concat(prediction_frames, ignore_index=True)


def fit_stacking_model(oof_predictions, model_names):
    """Fit a nonnegative linear stacker to out-of-fold base-model predictions."""
    stacker = LinearRegression(positive=True)
    stacker.fit(oof_predictions[model_names], oof_predictions["actual"])
    return stacker


def fit_model_collection(training_data, features, estimators, target, verbose=True):
    fitted_models = {}
    for model_name, estimator in estimators.items():
        if verbose:
            print(f"Fitting {model_name}")
        fitted_model = clone(estimator)
        selected_estimator = fitted_model.named_steps.get("model")
        if hasattr(selected_estimator, "n_jobs"):
            selected_estimator.set_params(n_jobs=-1)
        fitted_model.fit(training_data[features], training_data[target])
        if hasattr(selected_estimator, "n_jobs"):
            # Parallel fitting is useful, but repeated small prediction jobs are
            # faster and more portable without creating a worker pool each time.
            selected_estimator.set_params(n_jobs=1)
        fitted_models[model_name] = fitted_model
    return fitted_models


def predict_model_collection(data, features, fitted_models):
    return pd.DataFrame(
        {name: model.predict(data[features]) for name, model in fitted_models.items()},
        index=data.index,
    )


def compare_base_models_and_stack(selection_data, features, fitted_models, stacker, target):
    base_predictions = predict_model_collection(selection_data, features, fitted_models)
    rows = [
        {
            "model": model_name,
            "selection_mae": mean_absolute_error(selection_data[target], prediction),
        }
        for model_name, prediction in base_predictions.items()
    ]
    stacked_prediction = stacker.predict(base_predictions[list(fitted_models)])
    rows.append(
        {
            "model": "Stacking Ensemble",
            "selection_mae": mean_absolute_error(selection_data[target], stacked_prediction),
        }
    )
    comparison = pd.DataFrame(rows).sort_values("selection_mae").reset_index(drop=True)
    predictions = base_predictions.copy()
    predictions["Stacking Ensemble"] = stacked_prediction
    return comparison, predictions


def build_model_bundle(
    training_data,
    features,
    margin_estimators,
    total_estimators,
    margin_stacker,
    total_stacker,
    margin_primary,
    total_primary,
    verbose=True,
):
    """Fit every base model and package the two selected prediction systems."""
    return {
        "features": list(features),
        "margin_models": fit_model_collection(
            training_data, features, margin_estimators, "actual_margin", verbose
        ),
        "total_models": fit_model_collection(
            training_data, features, total_estimators, "actual_total", verbose
        ),
        "margin_stacker": margin_stacker,
        "total_stacker": total_stacker,
        "margin_model_names": list(margin_estimators),
        "total_model_names": list(total_estimators),
        "margin_primary": margin_primary,
        "total_primary": total_primary,
    }


def refit_model_bundle(training_data, saved_bundle, verbose=True):
    """Refit saved base-model specifications on all currently completed games."""
    features = saved_bundle["features"]
    margin_estimators = {
        name: clone(model) for name, model in saved_bundle["margin_models"].items()
    }
    total_estimators = {
        name: clone(model) for name, model in saved_bundle["total_models"].items()
    }
    return build_model_bundle(
        training_data=training_data,
        features=features,
        margin_estimators=margin_estimators,
        total_estimators=total_estimators,
        margin_stacker=saved_bundle["margin_stacker"],
        total_stacker=saved_bundle["total_stacker"],
        margin_primary=saved_bundle["margin_primary"],
        total_primary=saved_bundle["total_primary"],
        verbose=verbose,
    )


def _safe_model_name(model_name):
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("support_vector_regression", "svr")
    )


def _target_predictions(data, model_bundle, target):
    models = model_bundle[f"{target}_models"]
    names = model_bundle[f"{target}_model_names"]
    base_predictions = predict_model_collection(data, model_bundle["features"], models)
    stacked_prediction = model_bundle[f"{target}_stacker"].predict(base_predictions[names])
    all_predictions = base_predictions.copy()
    all_predictions["Stacking Ensemble"] = stacked_prediction
    primary = model_bundle[f"{target}_primary"]
    return all_predictions, all_predictions[primary].to_numpy()


def add_score_predictions(data, model_bundle, prefix="model"):
    output = data.copy()
    if "margin_models" not in model_bundle:  # Compatibility with the first project version.
        features = model_bundle["features"]
        output[f"{prefix}_margin"] = model_bundle["margin_model"].predict(output[features])
        output[f"{prefix}_total"] = model_bundle["total_model"].predict(output[features])
    else:
        margin_predictions, selected_margin = _target_predictions(output, model_bundle, "margin")
        total_predictions, selected_total = _target_predictions(output, model_bundle, "total")
        for model_name in margin_predictions:
            output[f"{prefix}_{_safe_model_name(model_name)}_margin"] = margin_predictions[
                model_name
            ].to_numpy()
        for model_name in total_predictions:
            output[f"{prefix}_{_safe_model_name(model_name)}_total"] = total_predictions[
                model_name
            ].to_numpy()
        output[f"{prefix}_margin"] = selected_margin
        output[f"{prefix}_total"] = selected_total
    output[f"{prefix}_home_score"] = np.maximum(
        0, (output[f"{prefix}_total"] + output[f"{prefix}_margin"]) / 2
    )
    output[f"{prefix}_away_score"] = np.maximum(
        0, (output[f"{prefix}_total"] - output[f"{prefix}_margin"]) / 2
    )
    return output


def fit_best_models(training_data, features, margin_estimator, total_estimator):
    """Retained for compatibility with notebooks created before the ensemble extension."""
    margin_model = clone(margin_estimator).fit(training_data[features], training_data["actual_margin"])
    total_model = clone(total_estimator).fit(training_data[features], training_data["actual_total"])
    return {"margin_model": margin_model, "total_model": total_model, "features": list(features)}


def add_prediction_intervals(predictions, calibration_predictions, coverage=0.90, prefix="model"):
    output = predictions.copy()
    margin_error = np.abs(
        calibration_predictions["actual_margin"] - calibration_predictions[f"{prefix}_margin"]
    )
    total_error = np.abs(
        calibration_predictions["actual_total"] - calibration_predictions[f"{prefix}_total"]
    )
    margin_radius = float(margin_error.quantile(coverage))
    total_radius = float(total_error.quantile(coverage))
    output[f"{prefix}_margin_low"] = output[f"{prefix}_margin"] - margin_radius
    output[f"{prefix}_margin_high"] = output[f"{prefix}_margin"] + margin_radius
    output[f"{prefix}_total_low"] = output[f"{prefix}_total"] - total_radius
    output[f"{prefix}_total_high"] = output[f"{prefix}_total"] + total_radius
    return output


def save_model_bundle(model_bundle, path):
    joblib.dump(model_bundle, path)


def load_model_bundle(path):
    return joblib.load(path)
