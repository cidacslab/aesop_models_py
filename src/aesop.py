import pandas as pd
import polars as pl


def evi(
    series: pl.Series | pd.Series,
    m: int = 5,
    r_a: int = 4,
    mu: int = 8,
    c: float | int = 0.2
):
    series = pl.Series(series).alias("series")
    df = pl.DataFrame(series)
    nm_series = "series"

    df = df.with_columns(
            pl.col(nm_series)
            .rolling_mean(window_size=r_a)
            .fill_nan(0)
            .fill_null(0)
            .alias("tseries")
        ).with_columns(
            pl.col("tseries")
            .rolling_std(window_size=m)
            .alias("std_windows")
        ).with_columns((
                (pl.col("std_windows") - pl.col("std_windows").shift())
                / abs(pl.col("std_windows").shift())
            ).fill_null(0).round(2).alias("evi_t1_t")
        ).with_columns(
            pl.when(pl.col("evi_t1_t").is_infinite())
            .then(pl.lit(0))
            .otherwise(pl.col("evi_t1_t"))
            .alias("evi_t1_t")
        ).with_columns(
            pl.col(nm_series)
            .rolling_mean(window_size=mu)
            .fill_null(0)
            .alias("mean_windows")
        ).with_columns(
            pl.when(
                (pl.col(nm_series) >= 0.75 * pl.col("mean_windows")) &
                (pl.col(nm_series) < pl.col("mean_windows"))
            ).then(0.5)
            .when(
                pl.col(nm_series) >= pl.col("mean_windows")
            ).then(1).otherwise(0).alias("cat_mu")
        ).with_columns(
            (pl.col("evi_t1_t") >= c).cast(pl.Int8).alias("cat_evi")
        ).with_columns(
            pl.when(
                (pl.col("cat_evi") == 1) & (pl.col("cat_mu") == 1)
            ).then(1).otherwise(0).alias("ind")
        ).with_columns(
            pl.when(
                (pl.col("evi_t1_t") > c) & (pl.col("cat_mu") == 1)
            ).then(1).otherwise(0).alias("ind_2")
        ).with_columns(
            (
                (pl.col(nm_series) - pl.col("mean_windows")) /
                ((1 + c) * pl.col("std_windows").shift())
            ).alias("k_opt")
        ).with_columns(
            (
                pl.col("tseries") +
                pl.col("k_opt").abs() *
                (1 + c) *
                pl.col("std_windows").shift()
            ).fill_nan(0).fill_null(0).cast(pl.Int32).alias("upperbound")
        ).with_columns(
            pl.when(
                pl.col(nm_series) > pl.col("upperbound")
            ).then(1).otherwise(0).alias("alarm")
        )
    return df.select("alarm").to_series().to_list(), df.select("upperbound").to_series().to_list()
