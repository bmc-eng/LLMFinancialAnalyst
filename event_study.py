import bql

from bloomberg.bquant.signal_lab.workflow.node import (
    industry_grouping, portfolio_construction)
from bloomberg.bquant.signal_lab.signal.transformers import WeightingScheme
from bloomberg.bquant.signal_lab.workflow.factory import (
    UniverseFactory,
    DataItemFactory,
    SignalFactory,
)
from bloomberg.bquant.signal_lab.workflow import (
    AnalyticsDataConfig,
    build_backtest,
)

from bloomberg.bquant.signal_lab.workflow.utils import get_sandbox_path
from bloomberg.bquant.signal_lab.workflow.workflow_orchestrator import _WorkflowResults

import utils.event_backtest_helper as ebh
from backtest_params import get_universe_params, get_return_params, get_analytics_data_config

import numpy as np
import pandas as pd


def build_port_weights(signal: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Function to convert a dataframe of signals into a portfolio by adjusting the weights
    signal: DataFrame of a pricing signal to use as the base
    events_df: DataFrame with Date, Security, Decision (BUY/ SELL/ HOLD) and Confidence Columns
    OUTPUT: Dataframe of the weights of a long/ short portfolio
    """
    long_portfolio =  signal.copy(deep=True)
    short_portfolio = signal.copy(deep=True)
    
    long_portfolio.loc[:,:] = False
    short_portfolio.loc[:,:] = False
    signal.loc[:,:] = 1
    
    # STEP 1 get the list of securities in th df_events database
    unique_securities = list(events_df['Security'].unique())
    
    # STEP 2: iterate over the list of securities to look at the individual trades
    for security in unique_securities:
        try:
            security_trades = events_df[events_df['Security'] == security]
        
            # STEP 3: iterate over the trades and update the long/ short portfolio depending on trade direction
            for row in security_trades.itertuples():
                if row.Decision == 'BUY':
                    long_portfolio[security].loc[row.Date:] = True
                    short_portfolio[security].loc[row.Date:] = False
                if row.Decision == 'SELL':
                    long_portfolio[security].loc[row.Date:] = False
                    short_portfolio[security].loc[row.Date:] = True
                if row.Decision == 'HOLD':
                    continue
                if row.Decision == 'Missing':
                    continue
        except KeyError:
            print(f"Missing: {security}")
    
    # STEP 4: create an equal weighted long and short leg
    long_portfolio_leg = ebh.leg_portfolio(
        signal=signal,
        weighting_scheme=WeightingScheme.EQUAL,
        assets_filter=long_portfolio,
        long_leg=True
    )
    
    short_portfolio_leg = ebh.leg_portfolio(
        signal=signal,
        weighting_scheme=WeightingScheme.EQUAL,
        assets_filter=short_portfolio,
        long_leg=False
    )
    
    long_short_portfolio = long_portfolio_leg.add(
        short_portfolio_leg,
        fill_value=0.0,
    )
    
    # STEP 5: return the long and short portfolios
    return long_short_portfolio

def signal_fn(signal: DataItemFactory) -> DataItemFactory:
        return signal

class EventBacktest:

    def __init__(self, start: str, end: str, universe_name: str, data_pack_path: str):
        """ Initialise the Backtester with time period and universe"""
        self.start: str         = start
        self.end: str           = end
        self.universe_name: str = universe_name
        self.bq: bql.Service    = bql.Service()

        # Load the datasets from the datapacks
        self.universe, self.benchmark, self.trading_calendar = get_universe_params(
            self.start, self.end, self.universe_name, data_pack_path
        )
        self.price, self.cur_mkt_cap, self.total_return = get_return_params(
            self.start, self.end, data_pack_path
        )
        self.analytics_data_config = get_analytics_data_config(
            self.start, self.end, self.universe_name, data_pack_path
        )


    def _bql_execute_single(self, univ: list[str], field: dict[str, bql.om.bql_item.BqlItem]) -> pd.DataFrame:
        """Execute a BQL query with a universe and one field"""
        
        req = bql.Request(univ, field)
        data = self.bq.execute(req)
        return data[0].df()


    def _convert_to_figi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to convert Bloomberg tickers in a dataframe to FIGIs for ESL"""
        
        univ      = df['Security'].to_list()
        field     = {'figi': self.bq.data.composite_id_bb_global()}
        figi      = self._bql_execute_single(univ, field)
        merged_df = df.merge(figi, left_on='Security', right_index=True).sort_index()
        return merged_df[['Date', 'figi', 'Decision', 'Confidence']].rename(columns={'figi':'Security'}) 

    
    def run(self, events_df: pd.DataFrame, run_name: str) -> _WorkflowResults:
        """Execute an Events backtest using a dataframe of trade events
        events_df: Dataframe of trades with Date, Security, Trade direction (BUY/ SELL/ HOLD) and Confidence score
        run_name: Name for the backtest
        """
        # STEP 1: Convert the events_df securities into FIGI
        events_figi_df = self._convert_to_figi(events_df)

        # STEP 2: Create the pricing signal used as dummy input into ESL
        self.price.bind_universe(self.universe)
        price_df = self.price.df()
        signal = SignalFactory.from_user(
            user_func=signal_fn,
            start=self.start,
            end= self.end,
            label=run_name,
            signal=self.total_return
        )
        
        # STEP 3: Construct the portfolio weighting scheme
        trading_dates = list(events_df['Date'].unique())
        port_long_short = portfolio_construction.from_user(
            compute_weights_fn= build_port_weights,
            total_returns=self.total_return,
            trading_calendar=self.trading_calendar,
            implementation_lag=1,
            rebalance_freq="D",
            #trading_dates=trading_dates,
            events_df=events_figi_df,
            signal=price_df,
        )

        # STEP 4: Build the backtest
        backtest = build_backtest(
            universe=self.universe,                                  # Univ of choice from DataPack
            benchmark_universe=self.benchmark,                       # Benchmark of choice from DataPack 
            start=self.start,                                        # Backtest start date
            end=self.end,                                            # Backtest end date
            namespace='events-bt',                 # The user S3 sandbox storage 
            signals=[signal],                                 # My list of signals to use
        
            portfolio_construction = port_long_short,
        
            reports=[
                "PerformanceReport",
                "QuantileAnalyticsReport",
                "DescriptiveStatisticsReport",
            ],
            analytics_data_config= self.analytics_data_config,
        )
        # STEP 5: Run and return the results
        return backtest.evaluate_graph()