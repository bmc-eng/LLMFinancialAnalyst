import os
import bql
import logging
import json

from ipywidgets import IntProgress
from IPython.display import display
from tqdm import tqdm
import pandas as pd

bq = bql.Service()

logging.basicConfig(format='%(asctime)s %(message)s', filename="Data/datarequest.log")


def setup_request(universe, as_of_date):
    """
    Set up the BQL query with the correct fields for financial statements, 
    stock price and metadata for the sector. 
    """

    univ = bq.univ.list(universe, dates=as_of_date)
    params = {
        'currency': 'USD',
        'fa_period_type': 'LTM',
        'fa_period_offset': bq.func.range('-5Q','0Q'),
        'fa_period_year_end': 'C1231',
        'dates': as_of_date,
        'fa_act_est_data': 'A',
        'fa_period_type_source': 'Q'
    }

    params_no_currency = {
        'fa_period_type': 'LTM',
        'fa_period_offset': bq.func.range('-5Q','0Q'),
        'fa_period_year_end': 'C1231',
        'dates': as_of_date,
        'fa_act_est_data': 'A',
        'fa_period_type_source': 'Q'
    }

    is_fields = {
        '01 Revenue (Adj)': bq.data.sales_rev_turn(**params),
        '02 Sales and Services Revenues (Adj)': bq.data.is_s_and_sr_gaap(**params),
        '03 Financing Revenue (Adj)': bq.data.is_financing_revenue_gaap(**params),
        '04 Other Revenue (Adj)': bq.data.is_other_revenue_gaap(**params),
        '05 Cost of Revenue (Adj)': bq.data.is_cogs_to_fe_and_pp_and_g(**params),
        '06 Cost of Goods & Services Sold (Adj)': bq.data.is_cog_and_ss_gaap(**params),
        '07 Cost of Financing Revenue (Adj)': bq.data.is_cost_of_financing_rev_gaap(**params),
        '08 Gross Profit (Adj)': bq.data.gross_profit(**params),
        '09 Other Operating Income (Adj)': bq.data.is_other_oper_inc(**params),
        '10 Operating Expenses (Adj)': bq.data.is_operating_expn(**params),
        '11 Selling, General and Administrative Expense (Adj)': bq.data.is_sg_and_a_expense(**params),
        '12 R&D Expense Adjusted (Adj)': bq.data.is_opex_r_and_d_gaap(**params),
        '13 Other Operating Expenses (Adj)': bq.data.is_other_operating_expenses_gaap(**params),
        '14 Operating Income or Losses (Adj)': bq.data.is_oper_inc(**params),
        '15 Non-Operating (Income) Loss (Adj)': bq.data.is_non_operating_inc_loss_gaap(**params),
        '16 Net Interest Expense (Adj)': bq.data.is_net_interest_expense(**params),
        '17 Interest Expense (Adj)': bq.data.is_int_expense(**params),
        '18 Interest Income (Adj)': bq.data.is_int_inc(**params),
        '19 Foreign Exch Losses (Gains) (Adj)': bq.data.is_foreign_exch_loss(**params),
        '20 Other Non-Operating (Income) Loss (Adj)': bq.data.is_other_nonop_inc_loss_gaap(**params),
        '21 Pretax Income (Loss), Adjusted (Adj)': bq.data.pretax_inc(**params),
        '22 Abnormal Losses (Gains)': bq.data.is_abnormal_item(**params),
        '23 Merger / Acquisition Expense': bq.data.is_merger_acquisition_expense(**params),
        '24 Sale of Business': bq.data.is_sale_of_business(**params),
        '25 Restructuring Expenses': bq.data.is_restructuring_charges(**params),
        '26 Gain/Loss on Investments': bq.data.is_gain_loss_on_investments(**params),
        '27 Other Abnormal Items': bq.data.is_other_one_time_items(**params),
        '28 Pretax Income (Loss), GAAP': bq.data.pretax_inc(**params),
        '29 Income Tax Expense (Benefit)': bq.data.is_inc_tax_exp(**params),
        '30 Current Income Tax': bq.data.is_current_income_tax_benefit(**params),
        '31 Deferred Income Tax': bq.data.is_deferred_income_tax_benefit(**params),
        '32 Income (Loss) from Continuing Operations': bq.data.is_inc_bef_xo_item(**params),
        '33 Net Extraordinary Losses (Gains)': bq.data.xo_gl_net_of_tax(**params),
        '34 Discontinued Operations': bq.data.is_discontinued_operations(**params),
        '35 Extraordinary Items and Accounting Changes': bq.data.is_extraord_items_and_acctg_chng(**params),
        '36 Net Income Including Minority Interest': bq.data.ni_including_minority_int_ratio(**params),
        '37 Net Income/Net Profit (Losses)': bq.data.net_income(**params),
        '38 Preferred Dividends': bq.data.is_tot_cash_pfd_dvd(**params),
        '39 Other Adjustments': bq.data.other_adjustments(**params),
        '40 Net Income Avail to Common, GAAP': bq.data.earn_for_common(**params),
        '41 Net Income Avail to Common, Adj (Adj)': bq.data.earn_for_common(**params),
        '42 Net Abnormal Losses (Gains)': bq.data.is_net_abnormal_items(**params),
        '43 Net Extraordinary Losses (Gains)': bq.data.xo_gl_net_of_tax(**params),
        '44 Basic Weighted Average Number of Shares': bq.data.is_avg_num_sh_for_eps(**params_no_currency),
        '45 Basic Earnings per Share': bq.data.is_eps(**params),
        '46 Basic EPS from Continuing Operations': bq.data.is_earn_bef_xo_items_per_sh(**params),
        '47 Basic EPS from Continuing Operations': bq.data.is_basic_eps_cont_ops(**params),
        '48 Diluted Weighted Average Shares': bq.data.is_sh_for_diluted_eps(**params_no_currency),
        '49 Diluted EPS': bq.data.is_diluted_eps(**params),
        '50 Diluted EPS from Continuing Operations': bq.data.is_dil_eps_bef_xo(**params),
        '51 Diluted EPS from Continuing Operations, Adj': bq.data.is_dil_eps_cont_ops(**params)
    }
    
    bs_fields = {
        '01 Cash, Cash Equivalents & STI' : bq.data.c_and_ce_and_sti_detailed(**params),
        '02 Cash & Cash Equivalents' : bq.data.bs_cash_near_cash_item(**params),
        '03 ST Investments' : bq.data.bs_mkt_sec_other_st_invest(**params),
        '04 Accounts & Notes Receiv' : bq.data.bs_acct_note_rcv(**params),
        '05 Inventories' : bq.data.bs_inventories(**params),
        '06 Raw Materials' : bq.data.invtry_raw_materials(**params),
        '07 Work In Process' : bq.data.invtry_in_progress(**params),
        '08 Finished Goods' : bq.data.invtry_finished_goods(**params),
        '09 Other Inventory' : bq.data.bs_other_inv(**params),
        '10 Other ST Assets' : bq.data.other_current_assets_detailed(**params),
        '11 Derivative & Hedging Assets' : bq.data.bs_deriv_and_hedging_assets_st(**params),
        '12 Discontinued Operations' : bq.data.bs_assets_of_discontinued_ops_st(**params),
        '13 Misc ST Assets' : bq.data.bs_other_cur_asset_less_prepay(**params),
        '14 Total Current Assets' : bq.data.bs_cur_asset_report(**params),
        '15 Property, Plant & Equip, Net' : bq.data.bs_net_fix_asset(**params),
        '16 Property, Plant & Equip' : bq.data.bs_gross_fix_asset(**params),
        '17 Accumulated Depreciation' : bq.data.bs_accum_depr(**params),
        '18 LT Investments & Receivables' : bq.data.bs_lt_invest(**params),
        '19 LT Receivables' : bq.data.bs_lt_receivables(**params),
        '20 Other LT Assets' : bq.data.bs_other_assets_def_chrg_other(**params),
        '21 Total Intangible Assets' : bq.data.bs_disclosed_intangibles(**params),
        '22 Goodwill' : bq.data.bs_goodwill(**params),
        '23 Other Intangible Assets' : bq.data.other_intangible_assets_detailed(**params),
        '24 Deferred Tax Assets' : bq.data.bs_deferred_tax_assets_lt(**params),
        '25 Derivative & Hedging Assets' : bq.data.bs_deriv_and_hedging_assets_lt(**params),
        '26 Prepaid Pension Costs' : bq.data.bs_prepaid_pension_costs_lt(**params),
        '27 Discontinued Operations' : bq.data.bs_assets_of_discontinued_ops_lt(**params),
        '28 Misc LT Assets' : bq.data.other_noncurrent_assets_detailed(**params),
        '29 Total Noncurrent Assets' : bq.data.bs_tot_non_cur_asset(**params),
        '30 Total Assets' : bq.data.bs_tot_asset(**params),
        '31 Payables & Accruals' : bq.data.acct_payable_and_accruals_detailed(**params),
        '32 Accounts Payable' : bq.data.bs_acct_payable(**params),
        '33 Accrued Taxes' : bq.data.bs_taxes_payable(**params),
        '34 Interest & Dividends Payable' : bq.data.bs_interest_and_dividends_payable(**params),
        '35 Other Payables & Accruals' : bq.data.bs_accrual(**params),
        '36 ST Debt' : bq.data.bs_st_borrow(**params),
        '37 ST Borrowings' : bq.data.short_term_debt_detailed(**params),
        '38 ST Finance Leases' : bq.data.st_capital_lease_obligations(**params),
        '39 ST Operating Leases' : bq.data.bs_st_operating_lease_liabs(**params),
        '40 Current Portion of LT Debt' : bq.data.bs_curr_portion_lt_debt(**params),
        '41 Other ST Liabilities' : bq.data.other_current_liabs_sub_detailed(**params),
        '42 Deferred Revenue' : bq.data.st_deferred_revenue(**params),
        '43 Derivatives & Hedging' : bq.data.bs_derivative_and_hedging_liabs_st(**params),
        '44 Discontinued Operations' : bq.data.bs_liabs_of_discontinued_ops_st(**params),
        '45 Misc ST Liabilities' : bq.data.other_current_liabs_detailed(**params),
        '46 Total Current Liabilities' : bq.data.bs_cur_liab(**params),
        '47 LT Debt' : bq.data.bs_lt_borrow(**params),
        '48 LT Borrowings' : bq.data.long_term_borrowings_detailed(**params),
        '49 LT Finance Leases' : bq.data.lt_capital_lease_obligations(**params),
        '50 LT Operating Leases' : bq.data.bs_lt_operating_lease_liabs(**params),
        '51 Other LT Liabilities' : bq.data.other_noncur_liabs_sub_detailed(**params),
        '52 Accrued Liabilities' : bq.data.bs_accrued_liabilities(**params),
        '53 Pension Liabilities' : bq.data.pension_liabilities(**params),
        '54 Deferred Revenue' : bq.data.lt_deferred_revenue(**params),
        '55 Derivatives & Hedging' : bq.data.bs_derivative_and_hedging_liabs_lt(**params),
        '56 Discontinued Operations' : bq.data.bs_liabs_of_discontinued_ops_lt(**params),
        '57 Misc LT Liabilities' : bq.data.other_noncurrent_liabs_detailed(**params),
        '58 Total Noncurrent Liabilities' : bq.data.non_cur_liab(**params),
        '59 Total Liabilities' : bq.data.bs_tot_liab2(**params),
        '60 Preferred Equity' : bq.data.bs_pfd_eqy(**params),
        '61 Share Capital & APIC' : bq.data.bs_sh_cap_and_apic(**params),
        '62 Common Stock' : bq.data.bs_common_stock(**params),
        '63 Additional Paid in Capital' : bq.data.bs_add_paid_in_cap(**params),
        '64 Treasury Stock' : bq.data.bs_amt_of_tsy_stock(**params),
        '65 Retained Earnings' : bq.data.bs_pure_retained_earnings(**params),
        '66 Other Equity' : bq.data.other_equity_ratio(**params),
        '67 Equity Before Minority Interest' : bq.data.eqty_bef_minority_int_detailed(**params),
        '68 Minority/Non Controlling Interest' : bq.data.minority_noncontrolling_interest(**params),
        '69 Total Equity' : bq.data.total_equity(**params),
        '70 Total Liabilities & Equity' : bq.data.tot_liab_and_eqy(**params)

    }
    
    price = {
        'Price' : bq.data.px_last(dates=bq.func.range('-12M', as_of_date), currency='USD', fill='prev')
    }

    metadata = {
        'sector': bq.data.bics_level_1_sector_name(),
        'figi': bq.data.composite_id_bb_global(),
        'name': bq.data.name()
        
    }
    
    return univ, is_fields, bs_fields, price

def setup_metadata():
    #univ = bq.univ.list(universe)
    metadata = {
        'sector': bq.data.bics_level_1_sector_name(),
        'figi': bq.data.composite_id_bb_global(),
        'full_name': bq.data.name()
        
    }

    return metadata



class FinancialDataRequester:
    """
    Class to request historical point-in-time datasets for use in the backtests
    """
    def __init__(self, index_id:str, dataset_name:str, 
                 rebalance_dates: list[str], reporting_frequency: str):
        """
        Constructor Method. Creates the data requestor object
        index_id: str - Bloomberg Index identifier
        dataset_name: str - Name of the dataset requested for ID purposes
        rebalance_dates: list - List of dates to rebalance the index
        reporting_period: str - reporting frequency. Can be A, Q
        """
        self.index_id = index_id
        self.dataset_name = dataset_name
        self.rebalance_dates = rebalance_dates
        self.reporting_period = reporting_frequency
        self._bq = bql.Service()
        

    def create_financial_dataset(self) -> dict:
        """
        Main function to create the security datasets. 
        Return: Returns a dictionary of dates representing the reporting dates
        Inside each date is a disctionary of securities and datasets contained in the dictionary.
        """
        rebalance_dates = self.get_rebalance_dates()
        all_data = self.get_financial_data(rebalance_dates)
        # request all metadata
        unique_securities = self._get_unique_securities(all_data)
        #request metadata
        meta = setup_metadata()
        ref_datasets = self._process_metadata(unique_securities, meta)
        
        return self._combine_metadata(all_data, ref_datasets)

    
    def get_rebalance_dates(self) -> pd.DataFrame:
        """
        Function to calculate the securities rebalancing on which dates. 
        This calculates point-in-time which securities report on which dates.
        """
        rebalance_requests = []
        progress = tqdm(total=len(self.rebalance_dates), position=0, leave=True)
        
        for date in self.rebalance_dates:
            # Request the companies reporting on the rebalance date
            rebalance_requests.append(self._get_reporting_dates_per_rebalance(date, self.index_id))
            # Update progress 
            progress.update()
        # Create a dataframe of the results
        df = pd.concat(rebalance_requests)
        # Aggregate by deates and securities with the reporting date.
        df_concat = df[['ID','AS_OF_DATE','PERIOD_END_DATE']].sort_values('PERIOD_END_DATE', ascending=True).drop_duplicates(subset=['ID','PERIOD_END_DATE'], keep='first')
        return df_concat.set_index(['AS_OF_DATE','ID']).sort_values(['AS_OF_DATE'])

    
    
    def get_financial_data(self, dates_and_securities: pd.DataFrame):
        """
        Function to loop through all of the dates and securities and request financial statement
        information on each date. 
        dates_and_securities: DataFrame - Dataframe generated by get_rebalance_dates
        """
        all_data = {}
        is_first = True
        dates = dates_and_securities.reset_index()['AS_OF_DATE'].unique()
        max_count = len(dates_and_securities.index.get_level_values(0).unique())
        progress = tqdm(total=max_count, position=0, leave=True)
        # Loop through each date and extract securities
        for date in dates:
            if is_first:
                is_first=False
            else:
                as_of_date = str(date)[0:10]
                securities = list(dates_and_securities.loc[as_of_date].reset_index()['ID'])
                univ, is_fields, bs_fields, price = setup_request(securities, as_of_date) 
                try:
                    df_is = self._process_single_date(univ, is_fields)
                    df_bs = self._process_single_date(univ, bs_fields)
                    df_px = self._process_single_date(univ, price)
                    #df_mt = self._process_single_date(univ, meta)
                    all_data[as_of_date] = self._convert_to_dict(securities, df_is, df_bs, df_px)
                except:
                    logging.info(f'DataRequest: Missing data - {as_of_date}')
                progress.update()
        progress.update()
        return all_data


        
    def _combine_metadata(self, all_data: dict, metadata: pd.DataFrame):
        """
        Internal function to combine the metadata with the financial statement data
        """
        for date in all_data.keys():
            for sec in all_data[date].keys():
                all_data[date][sec]['mt'] = {'name': metadata.loc[sec].full_name,
                                             'figi': metadata.loc[sec].figi,
                                             'sector': metadata.loc[sec].sector}
        return all_data
    
    def _get_reporting_dates_per_rebalance(self, date: str, index: str):
        """
        Internal function for the BQL Request for the reporting dates of companies in the rebalance period
        date: str - the as of date for point in time data
        index: str - Bloomberg Index ID
        Return: DataFrame of reporting periods and securities
        """
        # set up the BQL query to get the members of the index as of a specific date
        univ = self._bq.univ.members(index, dates=date)
        # Request one of the financial statement items to see what day it is reporting
        field = self._bq.data.sales_rev_turn(dates=bq.func.range('-5Y','0D'), fa_period_type=self.reporting_period)
        # Request the data
        req = bql.Request(univ, field)
        data = self._bq.execute(req)
        df = data[0].df().dropna()
        # Convert to Dataframe with just the reporting period
        return df.sort_values('PERIOD_END_DATE', ascending=True).reset_index().drop_duplicates(subset=['ID','PERIOD_END_DATE'], keep='first')    
    

    
    def _format_request_to_df(self, data, fields: dict) -> pd.DataFrame:
        """
        Internal Function to reformat the dataframe and anonymise the datasets (removing the years)
        data: BQL response object
        fields: dictionary of BQL fields
        """
        fields = list(fields.keys())
        df_all = [data[index].df()[data[index].df()['PERIOD_END_DATE'] != 0]
                      .pivot(columns='PERIOD_END_DATE', values=[fields[index]])
                      .fillna(0) 
                      for index in range(0,len(fields))]
        df2 = pd.concat(df_all, axis=1)
        df3 = df2.stack().transpose().stack().unstack(level=0).transpose().fillna(0)
        df4 = df3.loc[:, (df3 != 0).any(axis=0)]
        # Reformat the columns to remove dates
        if len(df4.columns) == 6:
            df5 = df4.set_axis(['t-5','t-4','t-3','t-2','t-1', 't'], axis='columns')
        else:
            df5 = df4.drop(columns=df4.columns[0:(len(df4.columns)-6)])
            df5 = df5.set_axis(['t-5','t-4','t-3','t-2','t-1', 't'], axis='columns')
        # Reverse the direction of the dataset
        df6 = df5[df5.columns[::-1]]
        return df6.loc[(df6!=0).any(axis=1)]
    

    def _get_unique_securities(self, data) -> list[str]:
        """
        InternalFunction to get all of the unique securities in a dataset
        Return: List of securities
        """
        secs = []
        for date in data.keys():
            for sec in data[date].keys():
                if sec not in secs:
                    secs.append(sec)
        return secs
    
    
    
    def _convert_to_dict(self, securities: list, 
                         df_is: pd.DataFrame, 
                         df_bs: pd.DataFrame, 
                         df_px: pd.DataFrame) -> dict:
        """
        Internal Function to format the JSON object storing all of the 
        financial datasets
        securities: list - list of securities for a specific date
        df_is: pd.DataFrame - Income Statement Dataframe
        df_bs: pd.DataFrame - Balance Sheet Dataframe
        df_px: pd.DataFrame - Stock price Dataframe
        Returns dict of the date 
        """
        date = {}
        for security in securities:
            # Convert DF to JSON
            data = {}
            df_is_sec = df_is.loc[security].to_json()
            df_bs_sec = df_bs.loc[security].to_json()
            df_px_sec = df_px.loc[security].set_index('DATE')[['Price']].to_json()
            
            # Convert to string and store
            data['is'] = json.dumps(df_is_sec)
            data['bs'] = json.dumps(df_bs_sec)
            data['px'] = json.dumps(df_px_sec)

            date[security] = data
        return date
    

    
    def _process_metadata(self, securities: list, fields: dict) -> pd.DataFrame:
        """
        Internal Function to process the company metadata/ reference data
        """
        req = bql.Request(securities, fields)
        data = self._bq.execute(req)
        df = [d.df() for d in data]
        df1 = pd.concat(df, axis=1)
        return df1
        
    
    def _process_single_date(self, securities: list, fields: dict) -> pd.DataFrame:
        """
        Internal Function to process the financial statement and stock price
        BQL requests
        """
        req = bql.Request(securities, fields)
        data = self._bq.execute(req)
        if len(fields) > 1:
            # Send the dataframe to be anonymised and converted into the correct format
            return self._format_request_to_df(data, fields)
        else:
            return data[0].df()
