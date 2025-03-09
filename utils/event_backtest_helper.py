# This is from Bloomberg ESL documentation and is needed to construction long/ short leg in a portfolio

import pandas as pd
import numpy as np
from typing import Union, Callable
from inspect import signature

from bloomberg.bquant.signal_lab.signal.transformers.portfolio_construction import (  # noqa
    WeightingScheme,
)


def leg_portfolio(
    signal: pd.DataFrame,
    weighting_scheme: Union[WeightingScheme, Callable],
    assets_filter: pd.DataFrame,
    long_leg: bool,
    **kwargs: object,
) -> pd.DataFrame:
    if long_leg:
        weight_sign = 1
    else:
        weight_sign = -1

    raw_portfolio_leg = signal.where(assets_filter, np.nan)
    if callable(weighting_scheme):
        # use provided function to calculate weight
        try:
            if "signal" in signature(weighting_scheme).parameters:
                kwargs["signal"] = signal
            if "long_leg" in signature(weighting_scheme).parameters:
                kwargs["long_leg"] = long_leg
            weighted_portfolio = weighting_scheme(**kwargs)
            weighted_portfolio_leg = (
                weighted_portfolio.where(
                    assets_filter,
                    np.nan,
                )
                .reindex_like(signal)
                .astype(float)
            )
            assert ((
                weighted_portfolio_leg >= 0) | weighted_portfolio_leg.isna()
            ).all(
                axis=None
            ), (
                "The weighting_scheme function returns dataframe with "
                "negative value, please make sure it only returns "
                "non-negative value or NaN as magnitude of the weight"
            )
        except Exception as e:
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Fail to compute using user provided weighting function, {e}"
            ) from e
    elif weighting_scheme == WeightingScheme.EQUAL:
        # assign a weight of 1 for the selected assets in the case of `EQUAL`
        # weighting scheme
        weighted_portfolio_leg = raw_portfolio_leg.mask(
            lambda x: ~np.isnan(x), other=1
        )
    elif weighting_scheme == WeightingScheme.PROPORTIONAL:
        weighted_portfolio_leg = raw_portfolio_leg
    else:  # pragma: no cover
        raise ValueError(f"unsupported WeightScheme: {weighting_scheme}")

    normalized_portfolio_leg = weighted_portfolio_leg.div(
        weight_sign * weighted_portfolio_leg.sum(axis=1),
        axis=0,
    )
    return normalized_portfolio_leg