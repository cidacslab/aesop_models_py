from typing import Any, Generator
import polars as pl
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft 
import statsmodels.api as sm
import statsmodels.formula.api as smf


def lags_combination_vars(
    dependent_variable: str,
    serie: str,
    lags_variable: range,
    formula_component=None
) -> Generator[str, Any, None]:
    lags_var_comb = [f'{serie}_lag_{lag}' for lag in lags_variable]
    cumulative_sums_var = [
        ' + '.join(lags_var_comb[:i])
        for i in range(1, len(lags_var_comb) + 1)
    ]
    if formula_component is not None:
        for terms in cumulative_sums_var:
            components = [terms]
            formula_components = formula_component + ' + ' + ' + '.join(filter(None, components))
            formula = f'{dependent_variable} ~ {formula_components}'
            yield formula
    else:
        for terms in cumulative_sums_var:
            components = [terms]
            formula_components = ' + '.join(filter(None, components))
            formula = f'{dependent_variable} ~ {formula_components}'
            yield formula


def fill_nan_values(df, serie):
    df[f'fitted_values_{serie}_only'] = np.nan
    df[f'residuals_{serie}_only'] = np.nan
    df[f'sigma_{serie}_only'] = np.nan
    df[f'sigma__{serie}_only'] = np.nan
    df[f'sigma_t_{serie}_only'] = np.nan
    df['limit'] = np.nan
    df[f'LCL_{serie}_only'] = np.nan
    df['alarm'] = np.nan
    df[f'out_of_limits_below_{serie}_only'] = np.nan
    return df


def find_best_formula(
    df: pd.DataFrame | pl.DataFrame,
    serie: str,
    formulas: list[str] | Generator[str]
) -> pd.DataFrame:
    df = df.to_pandas() if isinstance(df, pl.DataFrame) else df
    df = fill_nan_values(df, serie)
    model, best_aic = None, float('inf')
    alpha = 1
    w = 1
    for formula in formulas:
        try:
            alpha = 1
            model = smf.glm(
                formula=formula,
                data=df,
                family=sm.families.NegativeBinomial(alpha=alpha)
            ).fit()

            current_aic = model.aic

            if current_aic < best_aic:
                best_aic = current_aic

                df[f'fitted_values_{serie}_only'] = model.fittedvalues
                df[f'residuals_{serie}_only'] = model.resid_deviance
                df[f'sigma_{serie}_only'] = np.std(model.resid_deviance)
                df[f'sigma__{serie}_only'] = np.std(model.fittedvalues)
                df[f'sigma_t_{serie}_only'] = model.fittedvalues.rolling(window=5).std()
                df['limit'] = df[f'fitted_values_{serie}_only'] + w * df[f'sigma_t_{serie}_only']
                df[f'LCL_{serie}_only'] = df[f'fitted_values_{serie}_only'] - w * df[f'sigma_t_{serie}_only']
                df['alarm'] = (df[f'{serie}_4'] >= df['limit']).astype(int)
                df[f'out_of_limits_below_{serie}_only'] = (df[f'{serie}_4'] < df['limit']).astype(int)
        except:
            print('A vida não é um moranguinho')

    return df[['time_trend', serie, 'alarm', 'limit']]


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
        an = 2.0 / N * np.real(yf_filtered_best[1:N//2])
        bn = -2.0 / N * np.imag(yf_filtered_best[1:N//2])

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

        df = df.assign(S_t_fou = fourier_series_values)

        t = np.arange(len(df))

        threshold = 1e-6


        df['a0'] = a0
        for n, (a, b) in enumerate(zip(an, bn), start=1):
            if np.abs(a) >= threshold:
                df[f'cos_{n}'] = a * np.cos(2 * np.pi * n * t / N)
            if np.abs(b) >= threshold:
                df[f'sin_{n}'] = b * np.sin(2 * np.pi * n * t / N)

        df['Reconstructed'] = df['a0'] + df.filter(like='cos').sum(axis=1) + df.filter(like='sin').sum(axis=1)
        
        df = df.assign(Reconstructed_log = np.log(df.Reconstructed))
        
        df.Reconstructed_log = df.Reconstructed_log.fillna(0)
        
        v1 = df.loc[~df['Reconstructed_log'].isin([np.inf,-np.inf]),'Reconstructed_log'].max()
        v2 = df.loc[~df['Reconstructed_log'].isin([np.inf,-np.inf]),'Reconstructed_log'].min()
        df['Reconstructed_log'] = df['Reconstructed_log'].replace([np.inf, -np.inf], [v1, v2])
        
        return df
