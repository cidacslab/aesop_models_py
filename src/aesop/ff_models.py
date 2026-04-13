from typing import Any, Generator
import polars as pl
import pandas as pd
from numba import njit
import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import invgamma
import statsmodels.api as sm
import statsmodels.formula.api as smf


def lags_combination_vars(
    dependent_variable: str, serie: str, lags_variable: range, formula_component=None
) -> Generator[str, Any, None]:
    lags_var_comb = [f"{serie}_lag_{lag}" for lag in lags_variable]
    cumulative_sums_var = [
        " + ".join(lags_var_comb[:i]) for i in range(1, len(lags_var_comb) + 1)
    ]
    if formula_component is not None:
        for terms in cumulative_sums_var:
            components = [terms]
            formula_components = (
                formula_component + " + " + " + ".join(filter(None, components))
            )
            formula = f"{dependent_variable} ~ {formula_components}"
            yield formula
    else:
        for terms in cumulative_sums_var:
            components = [terms]
            formula_components = " + ".join(filter(None, components))
            formula = f"{dependent_variable} ~ {formula_components}"
            yield formula


def fill_nan_values(df, serie):
    df[f"fitted_values_{serie}_only"] = np.nan
    df[f"residuals_{serie}_only"] = np.nan
    df[f"sigma_{serie}_only"] = np.nan
    df[f"sigma__{serie}_only"] = np.nan
    df[f"sigma_t_{serie}_only"] = np.nan
    df["limit"] = np.nan
    df[f"LCL_{serie}_only"] = np.nan
    df["alarm"] = np.nan
    df[f"out_of_limits_below_{serie}_only"] = np.nan
    return df


def find_best_formula(
    df: pd.DataFrame | pl.DataFrame, serie: str, formulas: list[str] | Generator[str]
) -> pd.DataFrame:
    df = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    df = fill_nan_values(df, serie)
    model, best_aic = None, float("inf")
    alpha = 1
    w = 1
    for formula in formulas:
        try:
            alpha = 1
            model = smf.glm(
                formula=formula,
                data=df,
                family=sm.families.NegativeBinomial(alpha=alpha),
            ).fit()

            current_aic = model.aic

            if current_aic < best_aic:
                best_aic = current_aic

                df[f"fitted_values_{serie}_only"] = model.fittedvalues
                df[f"residuals_{serie}_only"] = model.resid_deviance
                df[f"sigma_{serie}_only"] = np.std(model.resid_deviance)
                df[f"sigma__{serie}_only"] = np.std(model.fittedvalues)
                df[f"sigma_t_{serie}_only"] = model.fittedvalues.rolling(window=5).std()
                df["limit"] = (
                    df[f"fitted_values_{serie}_only"] + w * df[f"sigma_t_{serie}_only"]
                )
                df[f"LCL_{serie}_only"] = (
                    df[f"fitted_values_{serie}_only"] - w * df[f"sigma_t_{serie}_only"]
                )
                df["alarm"] = (df[f"{serie}_4"] >= df["limit"]).astype(int)
                df[f"out_of_limits_below_{serie}_only"] = (
                    df[f"{serie}_4"] < df["limit"]
                ).astype(int)
        except:
            print("A vida não é um moranguinho")

    return df[["time_trend", serie, "alarm", "limit"]]


def fourier_series(t, a0, an, bn, N):
    result = a0
    for n, (a, b) in enumerate(zip(an, bn), start=1):
        result += a * np.cos(2 * np.pi * n * t / N) + b * np.sin(2 * np.pi * n * t / N)
    return result


def best_fft_reconstruction(data_values, period=52):
    S = np.array(data_values)

    yf = fft(S)

    best_S_reconstructed = None
    best_tsig = None
    yf_filtered_best = None

    for tsig in [0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08]:
        threshold = np.max(np.abs(yf)) * tsig

        yf_filtered = yf.copy()
        yf_filtered[np.abs(yf) < threshold] = 0

        S_reconstructed = ifft(yf_filtered).real

        if S_reconstructed.var() > 0.001:
            best_S_reconstructed = S_reconstructed
            best_tsig = tsig
            yf_filtered_best = yf_filtered
            break

    return best_S_reconstructed, best_tsig, yf_filtered_best


def harmonic(df: pd.DataFrame | pl.DataFrame, col: str) -> None | pd.DataFrame:
    df = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    if not df.shape[0]:
        return None

    data_values = df[col][0:52].to_numpy()
    _, _, yf_filtered_best = best_fft_reconstruction(data_values, period=52)

    if yf_filtered_best is not None:
        N = 52
        a0 = np.real(yf_filtered_best[0]) / N
        an = 2.0 / N * np.real(yf_filtered_best[1 : N // 2])
        bn = -2.0 / N * np.imag(yf_filtered_best[1 : N // 2])

        decimal_places = 6

        equation = f"S(t) = {a0:.{decimal_places}f}"
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= 1e-6 or np.abs(b) >= 1e-6:
                equation += f" + ({a:.{decimal_places}f}) * cos(2π * {n} * t / {N}) + ({b:.{decimal_places}f}) * sin(2π * {n} * t / {N})"

        time_points = np.arange(len(data_values))
        fourier_series_values = fourier_series(time_points, a0, an, bn, N)

        data_values = df[col].to_numpy()
        time_points = np.arange(len(data_values))
        fourier_series_values = fourier_series(time_points, a0, an, bn, N)

        df = df.assign(S_t_fou=fourier_series_values)

        t = np.arange(len(df))

        threshold = 1e-6

        df["a0"] = a0
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= threshold:
                df[f"cos_{n}"] = a * np.cos(2 * np.pi * n * t / N)
            if np.abs(b) >= threshold:
                df[f"sin_{n}"] = b * np.sin(2 * np.pi * n * t / N)

        df["Reconstructed"] = (
            df["a0"]
            + df.filter(like="cos").sum(axis=1)
            + df.filter(like="sin").sum(axis=1)
        )

        df = df.assign(Reconstructed_log=np.log(df.Reconstructed))

        df.Reconstructed_log = df.Reconstructed_log.fillna(0)

        v1 = df.loc[
            ~df["Reconstructed_log"].isin([np.inf, -np.inf]), "Reconstructed_log"
        ].max()
        v2 = df.loc[
            ~df["Reconstructed_log"].isin([np.inf, -np.inf]), "Reconstructed_log"
        ].min()
        df["Reconstructed_log"] = df["Reconstructed_log"].replace(
            [np.inf, -np.inf], [v1, v2]
        )

        return df


@njit
def exponential_model(t, beta, gamma, t0):
    return beta * np.exp(gamma * (t - t0))


@njit
def compute_log_likelihood(y, t, beta, gamma, sigma, t0):
    y_pred = exponential_model(t, beta, gamma, t0)
    res = -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - y_pred) ** 2) / sigma**2
    return res.sum()


def gibbs_sample_sigma(y, beta, gamma, t, t0, alpha1=4, alpha2=1):
    residuals = y - exponential_model(t, beta, gamma, t0)
    N = len(y)
    alpha_post = N / 2 + alpha1
    beta_post = np.sum(residuals**2) / 2 + alpha2
    return invgamma.rvs(a=alpha_post, scale=beta_post)


def metropolis_hastings(
    y,
    t,
    t0=0,
    n_samples=5000,
    burn_in=500,
    thin=5,
    beta_range=(0.01, 10),
    gamma_range=(-1, 1),
    proposal_sd=(0.1, 0.05),
):
    samples = []
    beta, gamma = 1.0, 0.0
    sigma = 1.0

    for _ in range(n_samples):
        beta_prop = np.clip(np.random.normal(beta, proposal_sd[0]), *beta_range)
        gamma_prop = np.clip(np.random.normal(gamma, proposal_sd[1]), *gamma_range)

        loglike_current = compute_log_likelihood(y, t, beta, gamma, sigma, t0)
        loglike_prop = compute_log_likelihood(y, t, beta_prop, gamma_prop, sigma, t0)

        accept_prob = np.exp(loglike_prop - loglike_current)
        if np.random.rand() < accept_prob:
            beta, gamma = beta_prop, gamma_prop

        sigma = gibbs_sample_sigma(y, beta, gamma, t, t0)

        samples.append((beta, gamma, sigma))

    samples = samples[burn_in::thin]
    return np.array(samples)


@njit
def metropolis_hastings_nb(
    y,
    t,
    t0=0,
    n_samples=5000,
    burn_in=500,
    thin=5,
    beta_low=0.01,
    beta_high=10.0,
    gamma_low=-1.0,
    gamma_high=1.0,
    prop_sd0=0.1,
    prop_sd1=0.05,
    alpha1=4.0,
    alpha2=1.0,
):
    samples = np.empty((n_samples, 3))
    beta = 1.0
    gamma = 0.0
    sigma = 1.0

    for i in range(n_samples):
        # ====== propostas com clip manual ======
        # beta
        beta_prop = np.random.normal(beta, prop_sd0)
        if beta_prop < beta_low:
            beta_prop = beta_low
        elif beta_prop > beta_high:
            beta_prop = beta_high

        # gamma
        gamma_prop = np.random.normal(gamma, prop_sd1)
        if gamma_prop < gamma_low:
            gamma_prop = gamma_low
        elif gamma_prop > gamma_high:
            gamma_prop = gamma_high

        # ====== passo MH ======
        ll_curr = compute_log_likelihood(y, t, beta, gamma, sigma, t0)
        ll_prop = compute_log_likelihood(y, t, beta_prop, gamma_prop, sigma, t0)

        if np.log(np.random.rand()) < (ll_prop - ll_curr):
            beta, gamma = beta_prop, gamma_prop

        # ====== Gibbs para sigma ======
        res = y - exponential_model(t, beta, gamma, t0)
        post_a = y.size * 0.5 + alpha1
        post_beta = (res**2).sum() * 0.5 + alpha2
        g = np.random.gamma(post_a, 1.0 / post_beta)
        sigma = 1.0 / g

        samples[i, 0] = beta
        samples[i, 1] = gamma
        samples[i, 2] = sigma

    # burn-in e thinning
    return samples[burn_in::thin]


def normalize_series(y):
    y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
    return y + 1  # Set minimum to 1


def sliding_window_pvalues(y, window=14):
    pvalues = []
    t = np.arange(len(y))
    y_norm = normalize_series(y)

    for start in range(len(y) - window + 1):
        t_win = t[start : start + window]
        y_win = y_norm[start : start + window]
        # samples = metropolis_hastings_nb(y_win, t_win, t0=t_win[0])
        samples = metropolis_hastings_nb(
            y_win,
            t_win,
            t0=t_win[0],
            n_samples=5000,
            burn_in=500,
            thin=5,
            beta_low=0.01,
            beta_high=10,
            gamma_low=-1,
            gamma_high=1,
            prop_sd0=0.1,
            prop_sd1=0.05,
            alpha1=4,
            alpha2=1,
        )
        gamma_samples = samples[:, 1]
        pval = 1 - np.mean(gamma_samples > 0)
        pvalues.append(pval)

    return np.array(pvalues)
