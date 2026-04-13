import pandas as pd
import polars as pl
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import friedmanchisquare

from ff_models import (
    find_best_formula,
    lags_combination_vars,
    harmonic,
    sliding_window_pvalues,
)


warnings.filterwarnings("ignore")


def evi(
    series: pl.Series | pd.Series,
    m: int = 5,
    r_a: int = 4,
    mu: int = 8,
    c: float | int = 0.2,
):
    series = pl.Series(series).alias("series")
    df = pl.DataFrame(series)
    nm_series = "series"

    df = (
        df.with_columns(
            pl.col(nm_series)
            .rolling_mean(window_size=r_a)
            .fill_nan(0)
            .fill_null(0)
            .alias("tseries")
        )
        .with_columns(pl.col("tseries").rolling_std(window_size=m).alias("std_windows"))
        .with_columns(
            (
                (pl.col("std_windows") - pl.col("std_windows").shift())
                / abs(pl.col("std_windows").shift())
            )
            .fill_null(0)
            .round(2)
            .alias("evi_t1_t")
        )
        .with_columns(
            pl.when(pl.col("evi_t1_t").is_infinite())
            .then(pl.lit(0))
            .otherwise(pl.col("evi_t1_t"))
            .alias("evi_t1_t")
        )
        .with_columns(
            pl.col(nm_series)
            .rolling_mean(window_size=mu)
            .fill_null(0)
            .alias("mean_windows")
        )
        .with_columns(
            pl.when(
                (pl.col(nm_series) >= 0.75 * pl.col("mean_windows"))
                & (pl.col(nm_series) < pl.col("mean_windows"))
            )
            .then(0.5)
            .when(pl.col(nm_series) >= pl.col("mean_windows"))
            .then(1)
            .otherwise(0)
            .alias("cat_mu")
        )
        .with_columns((pl.col("evi_t1_t") >= c).cast(pl.Int8).alias("cat_evi"))
        .with_columns(
            pl.when((pl.col("cat_evi") == 1) & (pl.col("cat_mu") == 1))
            .then(1)
            .otherwise(0)
            .alias("ind")
        )
        .with_columns(
            pl.when((pl.col("evi_t1_t") > c) & (pl.col("cat_mu") == 1))
            .then(1)
            .otherwise(0)
            .alias("ind_2")
        )
        .with_columns(
            (
                (pl.col(nm_series) - pl.col("mean_windows"))
                / ((1 + c) * pl.col("std_windows").shift())
            ).alias("k_opt")
        )
        .with_columns(
            (
                pl.col("tseries")
                + pl.col("k_opt").abs() * (1 + c) * pl.col("std_windows").shift()
            )
            .fill_nan(0)
            .fill_null(0)
            .cast(pl.Int32)
            .alias("upperbound")
        )
        .with_columns(
            pl.when(pl.col(nm_series) > pl.col("upperbound"))
            .then(1)
            .otherwise(0)
            .alias("alarm")
        )
    )
    return df.select("alarm").to_series().to_list(), df.select(
        "upperbound"
    ).to_series().to_list()


def compute_negbi_spc(serie: pl.Series):
    data: list[pl.Series] = []
    serie_4 = serie.rolling_mean(window_size=4, min_samples=1).alias(f"{serie.name}_4")
    data.extend([serie, serie_4])
    for lag in range(1, 5):
        data.append(serie.shift(lag).alias(f"{serie.name}_lag_{lag}"))

    df = pl.DataFrame().with_columns(
        *[pl.Series(name=column.name, values=column.to_numpy()) for column in data]
    )
    df = df.with_row_index("time_trend")

    model = smf.glm(
        formula=f"{serie_4.name} ~ time_trend",
        data=df.to_pandas(),
        family=sm.families.NegativeBinomial(alpha=1),
    ).fit()

    pred = model.predict(df.to_pandas()).to_numpy()
    df = df.with_columns(
        pl.lit(model.params["time_trend"]).alias(f"coef_negbi_{serie_4.name}"),
        pl.lit(model.bse["time_trend"]).alias(f"std_err_negbi_{serie_4.name}"),
        pl.lit(model.tvalues["time_trend"]).alias(f"z_negbi_{serie_4.name}"),
        pl.lit(model.pvalues["time_trend"]).alias(f"p_values_negbi_{serie_4.name}"),
        pl.lit(model.conf_int().loc["time_trend", 0]).alias(
            f"IC_low_negbi_{serie_4.name}"
        ),
        pl.lit(model.conf_int().loc["time_trend", 1]).alias(
            f"IC_high_negbi_{serie_4.name}"
        ),
        pl.Series(values=pred, name=f"trend_line_negbi_{serie_4.name}"),
    ).with_columns(
        (pl.col(serie_4.name) - pl.col(f"trend_line_negbi_{serie_4.name}")).alias(
            f"dtrend_{serie.name}_negbi"
        )
    )

    p = len(df) // 2
    rows = len(df) // p
    x = df.select(f"dtrend_{serie.name}_negbi").to_numpy()[: rows * p]
    data_matrix = np.reshape(x, (rows, p))
    test_stat, p_value = friedmanchisquare(*data_matrix.T)

    final = []
    df = df.with_columns(
        pl.lit(p_value).alias(f"p_value_{serie.name}_negbi_friedman"),
        pl.lit(test_stat).alias(f"test_stat_{serie.name}_negbi_friedman"),
    )

    df1 = df.filter(
        (pl.col(f"p_value_{serie.name}_negbi_friedman") >= 0.05)
        & (pl.col(f"p_values_negbi_{serie_4.name}") >= 0.05)
    )

    formulas_df1 = lags_combination_vars(serie_4.name, serie.name, range(1, 4))
    df1 = pl.from_pandas(find_best_formula(df1, serie.name, formulas_df1))
    if df1.shape[0]:
        final.append(df1)

    df2 = df.filter(
        (pl.col(f"p_value_{serie.name}_negbi_friedman") >= 0.05)
        & (pl.col(f"p_values_negbi_{serie_4.name}") < 0.05)
    )

    formulas_df2 = lags_combination_vars(
        serie_4.name, serie.name, range(1, 4), formula_component="time_trend"
    )
    df2 = pl.from_pandas(find_best_formula(df2, serie.name, formulas_df2))
    if df2.shape[0]:
        final.append(df2)

    df3 = df.filter(
        (pl.col(f"p_value_{serie.name}_negbi_friedman") < 0.05)
        & (pl.col(f"p_values_negbi_{serie_4.name}") >= 0.05)
    )

    df3 = harmonic(df3, serie_4.name)
    if df3 is not None:
        formulas_df3 = lags_combination_vars(
            serie_4.name, serie.name, range(1, 4), formula_component="Reconstructed"
        )
        df3 = pl.from_pandas(find_best_formula(df3, serie.name, formulas_df3))
        if df3.shape[0]:
            final.append(df3)

    df4 = df.filter(
        (pl.col(f"p_value_{serie.name}_negbi_friedman") < 0.05)
        & (pl.col(f"p_values_negbi_{serie_4.name}") < 0.05)
    )

    df4 = harmonic(df4, serie_4.name)
    if df4 is not None:
        formulas_df4 = lags_combination_vars(
            serie_4.name,
            serie.name,
            range(1, 4),
            formula_component="time_trend + Reconstructed",
        )
        df4 = pl.from_pandas(find_best_formula(df4, serie.name, formulas_df4))
        if df4.shape[0]:
            final.append(df4)

    result = pl.concat(final)

    return result.select("alarm").to_series().to_list(), result.select(
        "limit"
    ).to_series().to_list()


def bayes(serie: list[int]):
    nums = -137
    nums = (len(serie) * -1) if len(serie) < (nums * -1) else nums
    pvals = sliding_window_pvalues(serie[nums:], window=8)
    return (1 - pvals).tolist()
