import polars as pl
from matplotlib import pyplot as plt
from matplotlib import axes
from loguru import logger


def _prepare_simple_lift_plot_data(
    data: pl.DataFrame,
    prediction_col: str,
    target_col: str,
    weights_col: str | None,
    num_groups: int = 10,
    weighted_mean=True,
) -> pl.DataFrame:
    # TODO: Should data be sorted by the weighted prediction if weights_col is set?
    data = data.sort(by=prediction_col, descending=False)

    if weighted_mean and not weights_col:
        raise ValueError("'weights_col' parameter needs to be set when 'weighted_mean' is True!")

    if weights_col is None or weighted_mean is False:
        weights_col = "DUMMY_WEIGHTS"
        data = data.with_columns(pl.lit(1.0).alias(weights_col))

    data = data.with_columns(pl.col(weights_col).cum_sum().alias("SUMMED_WEIGHTS"))
    summed_weights = data[weights_col].sum()
    labels = [str(cut) for cut in range(0, num_groups)] + ["PLUS_ONE"]
    logger.debug(f"Labels: {labels}")
    breaks = [(cut * (summed_weights + 1)) / (num_groups) for cut in range(1, num_groups + 1)]
    logger.debug(f"Breaks: {breaks}")

    data = data.with_columns(
        data["SUMMED_WEIGHTS"].cut(breaks=breaks, labels=labels).alias("CUT_WEIGHTS"),
        (pl.col(target_col) * pl.col(weights_col).alias(target_col)),
        (pl.col(prediction_col) * pl.col(weights_col).alias(prediction_col)),
        pl.col(prediction_col).alias("ORIGINAL_PREDICTION"),
    )

    aggregations = [
        (pl.sum(target_col) / pl.sum(weights_col)).alias(target_col),
        (pl.sum(prediction_col) / pl.sum(weights_col)).alias(prediction_col),
        pl.concat_str(
            pl.min("ORIGINAL_PREDICTION").round(1),
            pl.lit(" - "),
            pl.max("ORIGINAL_PREDICTION").round(1),
        ).alias("SEGMENT_NAMES"),
        pl.sum(weights_col),
        pl.max("SUMMED_WEIGHTS"),
        pl.len().alias("NUM_OBS"),
    ]
    data = data.group_by("CUT_WEIGHTS").agg(*aggregations).sort("CUT_WEIGHTS")
    data = data.with_columns(
        pl.concat_str(pl.col("CUT_WEIGHTS"), pl.lit(": "), pl.col("SEGMENT_NAMES")).alias("SEGMENT_NAMES")
    )

    return data


def plot_simple_lift_plot(
    data: pl.DataFrame,
    prediction_col: str,
    target_col: str,
    weights_col: str | None,
    num_groups: int = 10,
    weighted_mean=True,
    label_font_size=8,
) -> axes.Axes:
    """
    Create a simple lift plot to compare predictions and actual data.
    """
    data = _prepare_simple_lift_plot_data(
        data=data,
        prediction_col=prediction_col,
        target_col=target_col,
        weights_col=weights_col,
        num_groups=num_groups,
        weighted_mean=weighted_mean,
    )
    logger.debug(data)

    ax: axes.Axes
    fig, ax = plt.subplots()
    ax.plot(data["SEGMENT_NAMES"], data[prediction_col], label="Prediction")
    ax.plot(data["SEGMENT_NAMES"], data[target_col], label="Actual")
    ax.legend()
    ax.grid()
    ax.set(xlabel="Prediction segments", ylabel="Pure Premium")
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_fontsize(label_font_size)
    ylab.set_fontsize(label_font_size)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment="right", size=10)

    ax2: axes.Axes = ax.twinx()
    ax2.set_ylabel("Exposure")
    ax2.yaxis.get_label().set_fontsize(label_font_size)
    if weights_col is None or weighted_mean is False:
        weights_col = "DUMMY_WEIGHTS"
        data = data.with_columns(pl.lit(1.0).alias(weights_col))
        ax2.set_ylabel("Data points")
    ax2.bar(data["SEGMENT_NAMES"], data[weights_col], alpha=0.2)

    return ax


def _prepare_double_lift_plot_data(
    data: pl.DataFrame,
    first_model_col: str,
    second_model_col: str,
    target_col: str,
    weights_col: str | None,
    num_groups: int = 10,
    weighted_mean=True,
) -> pl.DataFrame:
    # ACTUAL, and other columns
    data = data.with_columns(
        (pl.col(first_model_col) / pl.col(second_model_col)).alias("RATIO"),
    )

    if weights_col is None or weighted_mean is False:
        weights_col = "dummy_weights"
        data = data.with_columns(pl.lit(1.0).alias(weights_col))

    data = data.sort("RATIO")
    data = data.with_columns(pl.cum_sum(weights_col).alias("SUMMED_WEIGHTS"))
    summed_exposure = data[weights_col].sum()
    labels = [str(cut) for cut in range(0, num_groups)] + ["MEH"]
    breaks = [(cut * (summed_exposure + 1)) / (num_groups) for cut in range(1, num_groups + 1)]

    data = data.with_columns(
        data["SUMMED_WEIGHTS"].cut(breaks=breaks, labels=labels).alias("CUT_WEIGHTS"),
        (pl.col(target_col) * pl.col(weights_col).alias(target_col)),
        (pl.col(first_model_col) * pl.col(weights_col).alias(first_model_col)),
        (pl.col(second_model_col) * pl.col(weights_col).alias(second_model_col)),
        pl.col(first_model_col).alias("ORIGINAL_PREDICTION_1"),
        pl.col(second_model_col).alias("ORIGINAL_PREDICTION_2"),
    )

    aggregations = [
        (pl.sum(first_model_col) / pl.sum(weights_col)).alias(first_model_col),
        (pl.sum(second_model_col) / pl.sum(weights_col)).alias(second_model_col),
        pl.concat_str(
            pl.min("RATIO").round(1),
            pl.lit(" - "),
            pl.max("RATIO").round(1),
        ).alias("SEGMENT_NAMES"),
        pl.sum(weights_col),
        pl.max("SUMMED_EXPOSURE"),
        pl.len().alias("NUM_OBS"),
        pl.sum(target_col).alias("ACTUAL_ORIGINAL"),
    ]
    data = data.group_by("CUT_WEIGHTS").agg(*aggregations).sort("CUT_WEIGHTS")
    data = data.melt(id_vars="CUT_WEIGHTS", value_vars=[first_model_col, second_model_col, target_col])

    return data


def _prepare_obs_pred_by_feature_value(
    data: pl.DataFrame,
    feature_col: str,
    prediction_col: str,
    target_col: str,
    weights_col: str | None,
    weighted_mean: bool = True,
) -> pl.DataFrame:
    original_weights_col = weights_col
    if weighted_mean and not weights_col:
        raise ValueError("'weights_col' parameter needs to be set when 'weighted_mean' is True!")

    if weights_col is None or weighted_mean is False:
        weights_col = "DUMMY_WEIGHTS"
        data = data.with_columns(pl.lit(1.0).alias(weights_col))

    data = data.with_columns(
        (pl.col(target_col) * pl.col(weights_col).alias(target_col)),
        (pl.col(prediction_col) * pl.col(weights_col).alias(prediction_col)),
        pl.col(prediction_col).alias("ORIGINAL_PREDICTION"),
    )

    aggregations = [
        (pl.sum(target_col) / pl.sum(weights_col)).alias(target_col),
        (pl.sum(prediction_col) / pl.sum(weights_col)).alias(prediction_col),
        pl.sum(weights_col),
        pl.mean("ORIGINAL_PREDICTION"),
        pl.len().alias("NUM_OBS"),
    ]
    if original_weights_col is not None:
        aggregations = aggregations + [
            pl.sum(original_weights_col).alias("ORIGINAL_EXPOSURE"),
        ]
    data = data.group_by(feature_col).agg(*aggregations).sort(feature_col)

    return data


def plot_obs_pred_by_feature_value(
    data: pl.DataFrame,
    feature_col: str,
    prediction_col: str,
    target_col: str,
    weights_col: str | None,
    weighted_mean: bool = True,
) -> axes.Axes:
    if weighted_mean is True and weights_col is None:
        raise ValueError("'weights_col' parameter needs to be set when 'weighted_mean' is True!")

    data = _prepare_obs_pred_by_feature_value(
        data=data,
        feature_col=feature_col,
        prediction_col=prediction_col,
        target_col=target_col,
        weights_col=weights_col,
        weighted_mean=weighted_mean,
    )

    ax: axes.Axes
    fig, ax = plt.subplots()
    ax.plot(data[feature_col], data[prediction_col], label="Prediction")
    ax.plot(data[feature_col], data[target_col], label="Actual")
    ax.legend()
    ax.grid()
    ax.set(xlabel=f"Values of: {feature_col}", ylabel="Pure Premium")
    ax.set_title(feature_col)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment="right", size=10)

    ax2: axes.Axes = ax.twinx()
    ax2.set_ylabel("Exposre")
    if weights_col is not None:
        ax2.bar(data[feature_col], data["ORIGINAL_EXPOSURE"], alpha=0.2)
    else:
        # TODO: What if weighted_mean is false? What should we show?
        ax2.bar(data[feature_col], data["NUM_OBS"], alpha=0.2)

    return fig
