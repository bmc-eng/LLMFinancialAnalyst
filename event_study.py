import bql

from bloomberg.bquant.signal_lab.workflow.node import (
    industry_grouping, portfolio_construction)
from bloomberg.bquant.signal_lab.signal.transformers import WeightingScheme
from bloomberg.bquant.signal_lab.workflow.factory import (
    UniverseFactory, DataItemFactory,SignalFactory,
)
from bloomberg.bquant.signal_lab.workflow import (
    AnalyticsDataConfig,
    build_backtest,
)

from bloomberg.bquant.signal_lab.data_workbench import (
    create_data_pack,
    load_data_pack,
    DataPack,
    DataPackConfig,
    StorageType,
    TradingCalendarProxy,
    FetchMode,
    FetchErrorHandling
)

from bloomberg.bquant.signal_lab.workflow.workflow_orchestrator import _WorkflowResults

import utils.event_backtest_helper as ebh
from utils.backtest_params import get_universe_params, get_return_params, get_analytics_data_config, get_item_definitions

import numpy as np
import pandas as pd


def build_port_weights(signal: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert a dataframe of signals into a portfolio by adjusting the weights
    
    signal: DataFrame of a pricing signal to use as the base
    events_df: DataFrame with Date, Security, Decision (BUY/ SELL/ HOLD) and Confidence Columns
    Returns: Dataframe of the weights of a long/ short portfolio
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
    """
    Function to return the dummy data signal during the backtest. 
    """
    return signal


class EventBacktest:
    """
    Event Backtesting Class. 
    """
    
    def __init__(self, start: str, end: str, universe_name: str, data_pack_path: str, reload_data:bool = False):
        """ Initialise the Backtester with time period and universe"""
        self.start: str         = start
        self.end: str           = end
        self.universe_name: str = universe_name
        self.bq: bql.Service    = bql.Service()
        self.bt_results: _WorkflowResults = None

        # load the datapack
        try:
            self._data_pack = load_data_pack(storage_path=data_pack_path, storage_type=StorageType.PARQUET)
        except ValueError: 
            print('Data missing')
            self._data_pack = self._create_new_datapack()

        
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
    
        # Set up the pricing signal
        self.price.bind_universe(self.universe)
        self.price_df = self.price.df()

        self.benchmark_id = f"IndexWeights['{self.universe_name}']"
        
            


    def _bql_execute_single(self, univ: list[str], 
                            field: dict[str, bql.om.bql_item.BqlItem]) -> pd.DataFrame:
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
        return merged_df[['Date', 'figi', 'Decision']].rename(columns={'figi':'Security'}) 

    
    def run(self, events_df: pd.DataFrame, 
            run_name: str, 
            use_trading_dts: bool = False,
            implementation_lag: int = 1,
            rebalance_freq: str = 'D') -> _WorkflowResults:
        """Execute an Events backtest using a dataframe of trade events
        events_df: Dataframe of trades with Date, Security, Trade direction (BUY/ SELL/ HOLD) and Confidence score
        run_name: Name for the backtest
        """
        # STEP 1: Convert the events_df securities into FIGI
        events_figi_df = self._convert_to_figi(events_df)

        # STEP 2: Create the pricing signal used as dummy input into ESL
        #price_df = self.price.df()
        signal = SignalFactory.from_user(
            user_func=signal_fn,
            start=self.start,
            end= self.end,
            label=run_name,
            signal=self.total_return
        )
        
        # STEP 3: Construct the portfolio weighting scheme
        if use_trading_dts:
            trading_dates = list(events_df['Date'].unique())
            port_long_short = portfolio_construction.from_user(
                compute_weights_fn= build_port_weights,
                total_returns=self.total_return,
                trading_calendar=self.trading_calendar,
                implementation_lag=implementation_lag,
                trading_dates=trading_dates,
                events_df=events_figi_df,
                signal=self.price_df,
            )
        else:
            port_long_short = portfolio_construction.from_user(
                compute_weights_fn= build_port_weights,
                total_returns=self.total_return,
                trading_calendar=self.trading_calendar,
                implementation_lag=implementation_lag,
                rebalance_freq=rebalance_freq,
                events_df=events_figi_df,
                signal=self.price_df,
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
        self.bt_results = backtest.evaluate_graph()
        return self.bt_results

    
    def get_return_data(self, results: _WorkflowResults = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the cumulative return datasets for the strategy and the benchmark
        Output: df_strategy_return, df_benchmark_return time-series """
        if self.bt_results == None and results == None:
            raise Exception("Backtest object is missing! Please call run() first to run the backtest")
        else:
            if results != None:
                self.bt_results = results
            # Return time series
            df_strategy_return = self.bt_results.analytics["CumulativeReturn"].read()['gross']['COMBINED'].read()
            # Benchmark Return
            df_benchmark_return = self.bt_results.analytics["CumulativeReturn"].read()['benchmark'][self.benchmark_id].read()
            return df_strategy_return, df_benchmark_return

    
    def get_performance_data(self, results: _WorkflowResults = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the YoY performance data for the benchmark and the strategy"""
        if self.bt_results == None and results == None:
            raise Exception("Backtest object is missing! Please call run() first to run the backtest")
        else:
            if results != None:
                self.bt_results = results
            # Strategy YoY Performance
            df_strategy_performance = self.bt_results.analytics['PerformanceStatisticsByYear'].read()['gross']['COMBINED'].read()
            # Benchmark YoY Performance
            df_benchmark_performance = self.bt_results.analytics['PerformanceStatisticsByYear'].read()['benchmark'][self.benchmark.id].read()
            return df_strategy_performance, df_benchmark_performance

    
    def _create_new_datapack(self):
        """
        Function to create a new datapack for backtest datasets if one does not exist
        """
        data_pack_config = DataPackConfig(pd.Timestamp(self.start),pd.Timestamp(self.end))
        data_pack_item_definitions = get_item_defintions(self.bq)

        universe_definitions = {
            index_name: BQLIndexUniverse(self.universe_name)
        }
        # create the datapack
        data_pack = create_data_pack(
            data_pack_config = data_pack_config,
            universe_definitions=universe_definitions,
            data_item_definitions=data_item_definitions,
            storage_type=StorageType.PARQUET,
            storage_path=datapack_path,
            overwrite_existing_data_pack=True  
        )

        # fetch the data
        data_pack.run_fetch()
        return data_pack
        


























            

    