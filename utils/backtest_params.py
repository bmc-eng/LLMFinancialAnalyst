
import bql 

from bloomberg.bquant.signal_lab.workflow.node import (
    industry_grouping, portfolio_construction)
from bloomberg.bquant.signal_lab.signal.transformers import WeightingScheme
from bloomberg.bquant.signal_lab.workflow.factory import (
    UniverseFactory,
    DataItemFactory,
    SignalFactory,
)

from bloomberg.bquant.signal_lab.data_workbench import (
    create_data_pack,
    load_data_pack,
    DataPack,
    DataPackConfig,
    StorageType,
    BQLIndexUniverse,
    BQLIndexWeight,
    BQLData,
    ListUniverse,
    TradingCalendarProxy,
    FetchMode,
    FetchErrorHandling
)

from bloomberg.bquant.signal_lab.workflow import AnalyticsDataConfig

def get_universe_params(start: str, end: str, universe_name: str, data_pack_path: str) -> tuple[UniverseFactory, DataItemFactory, DataItemFactory]:
    """Function to return the universe, benchmark and trading calendar information from DataPack"""
    # Create universe and benchmark using factory
    universe = UniverseFactory.from_data_pack(
        universe_name=universe_name, 
        data_pack_path=data_pack_path, 
        trading_calendar_data_item_id='trading_calendar',
        start=start,
        end=end,
        label="Universe",
    )
    
    benchmark = DataItemFactory.from_data_pack(
        universe_name=universe_name, 
        data_pack_path=data_pack_path, 
        data_item_id="index_weight",
        start=start,
        end=end,
    )
    
    trading_calendar = DataItemFactory.from_data_pack(
        data_pack_path=data_pack_path,
        data_item_id='trading_calendar',
        universe_name=universe_name,
        start=start,
        end=end,
    )
    
    return universe, benchmark, trading_calendar

def get_analytics_data_config(start: str, end: str, universe_name: str, data_pack_path: str) -> AnalyticsDataConfig:
    """Function to create the AnalyticsDataConfig object needed for the performance analytics in the backtest"""
    # bics mapping data
    bics_level_1 = DataItemFactory.from_data_pack(
        data_item_id="bics_level_1",
        data_pack_path=data_pack_path,
        date_column="ITERATION_DATE",
        start=start,
        end=end,
        use_universe_date_index=False
    )
    
    bics_level_2 = DataItemFactory.from_data_pack(
        data_item_id="bics_level_2",
        data_pack_path=data_pack_path,
        date_column="ITERATION_DATE",
        start=start,
        end=end,
        use_universe_date_index=False
    )
    
    bics_level_3 = DataItemFactory.from_data_pack(
        data_item_id="bics_level_3",
        data_pack_path=data_pack_path,
        date_column="ITERATION_DATE",
        start=start,
        end=end,
        use_universe_date_index=False
    )
    
    bics_level_4 = DataItemFactory.from_data_pack(
        data_item_id="bics_level_4",
        data_pack_path=data_pack_path,
        date_column="ITERATION_DATE",
        start=start,
        end=end,
        use_universe_date_index=False
    )
    
    industry_mapping_dict = {
        "BICS Sector Name Level 1": bics_level_1,
        "BICS Industry Group Level 2": bics_level_2,
        "BICS Industry Level 3": bics_level_3,
        "BICS Sub-Industry Level 4": bics_level_4,
    }
    
    bics_mapping = industry_grouping.build_industry_classification_mapping(**industry_mapping_dict)
    
    total_returns = DataItemFactory.from_data_pack(
        data_item_id="day_to_day_tot_return_gross_dvds",
        data_pack_path=data_pack_path, 
        date_column="DATE", 
        start=start, 
        end=end
    )
    
    trading_calendar = DataItemFactory.from_data_pack(
        data_pack_path=data_pack_path,
        data_item_id='trading_calendar',
        universe_name=universe_name,
        start=start,
        end=end,
    )
    
    beta = DataItemFactory.from_data_pack(
        data_item_id="beta", 
        data_pack_path=data_pack_path, 
        date_column="ITERATION_DATE", 
        start=start, 
        end=end, 
        sampling_frq="w", 
        fillna_method="ffill"
    )
    
    return AnalyticsDataConfig(
        beta=beta,
        industry_classification_mapping=bics_mapping,
        returns=total_returns,
        trading_calendar=trading_calendar,
    )


def get_return_params(start: str, end: str, data_pack_path: str) -> tuple[DataItemFactory, DataItemFactory, DataItemFactory]:
    """Function to return the returns/ performance datasets needed for calculating returns of the strategy"""
    price = DataItemFactory.from_data_pack(
        data_pack_path=data_pack_path,
        data_item_id="px_last",
        start=start,
        end=end,
        date_column="DATE",
        label="price",
    )
    
    total_returns = DataItemFactory.from_data_pack(
        data_item_id="day_to_day_tot_return_gross_dvds",
        data_pack_path=data_pack_path, 
        date_column="DATE", 
        start=start, 
        end=end
    )
    
    cur_mkt_cap = DataItemFactory.from_data_pack(
        data_item_id="cur_mkt_cap",
        data_pack_path=data_pack_path, 
        date_column="DATE", 
        start=start, 
        end=end
    )

    return price, cur_mkt_cap, total_returns

def get_analyst_params(start: str, end: str, data_pack_path: str) -> tuple[DataItemFactory, DataItemFactory]:
    """Function to return the analyst recommendations needed to construct the base strategy"""
    analyst_ratings = DataItemFactory.from_data_pack(
        data_item_id='analyst_rating',
        data_pack_path=data_pack_path, 
        date_column="DATE", 
        start=start, 
        end=end
    )
    return analyst_ratings
        

def get_item_definitions(bq: bql.Service):
    data_item_definitions = {
        "index_weight": BQLIndexWeight(),
        "trading_calendar": TradingCalendarProxy(),
    
        "analyst_rating": BQLData(bq.data.best_analyst_rating()),
        "target_price": BQLData(bq.data.best_target_price()),
    
        "cur_mkt_cap": BQLData(bq.data.cur_mkt_cap()),
        "day_to_day_tot_return_gross_dvds": BQLData(bq.data.day_to_day_tot_return_gross_dvds()),
        "beta": BQLData(bq.data.beta(), rolling_date=True, freq="w"),
        "px_last": BQLData(bq.data.px_last()),
    
        'bics_level_1': BQLData(bq.data.classification_name("bics", "1"), rolling_date=True),
        'bics_level_2': BQLData(bq.data.classification_name("bics", "2"), rolling_date=True),
        'bics_level_3': BQLData(bq.data.classification_name("bics", "3"), rolling_date=True),
        'bics_level_4': BQLData(bq.data.classification_name("bics", "4"), rolling_date=True),
    }
    return data_item_definitions








        