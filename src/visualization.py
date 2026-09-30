import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "navy": "#17324D",
    "blue": "#4C78A8",
    "orange": "#F28E2B",
    "green": "#59A14F",
    "red": "#E15759",
    "gray": "#8C8C8C",
}


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.dpi": 120,
        }
    )


def predicted_vs_actual(data, actual, predicted, title):
    figure, axis = plt.subplots()
    axis.scatter(data[actual], data[predicted], alpha=0.35, color=COLORS["blue"])
    limits = [
        min(data[actual].min(), data[predicted].min()),
        max(data[actual].max(), data[predicted].max()),
    ]
    axis.plot(limits, limits, linestyle="--", color=COLORS["red"], label="Perfect prediction")
    axis.set(xlabel="Actual", ylabel="Predicted", title=title)
    axis.legend()
    return figure, axis


def residual_plot(data, actual, predicted, title):
    residual = data[actual] - data[predicted]
    figure, axis = plt.subplots()
    axis.scatter(data[predicted], residual, alpha=0.35, color=COLORS["orange"])
    axis.axhline(0, linestyle="--", color="black")
    axis.set(xlabel="Predicted", ylabel="Actual - predicted", title=title)
    return figure, axis
