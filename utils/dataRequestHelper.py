import os
import bql

bq = bql.Service()

# S3 Helper Functionality
user_bucket_name = os.environ['BQUANT_SANDBOX_USER_BUCKET']
bqnt_username = os.environ['BQUANT_USERNAME']

def get_s3_folder():
    return f's3://{user_bucket_name}/{bqnt_username}/tmp/'

# set up the data requests
def setup_request(universe, as_of_date):
    #univ = bq.univ.members(universe, dates=as_of_date)

    univ = bq.univ.list(universe, dates=as_of_date)
    params = {
        'currency': 'USD',
        'fa_period_type': 'BT',
        'fa_period_offset': bq.func.range('-5Q','0Q'),
        'fa_period_year_end': 'C1231',
        'dates': as_of_date,
        'fa_act_est_data': 'A',
        'fa_period_type_source': 'Q'
    }

    params_no_currency = {
        'fa_period_type': 'BT',
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
    
    return univ, is_fields, bs_fields, price
