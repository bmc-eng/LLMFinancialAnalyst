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
    univ = bq.univ.members(universe, dates=as_of_date)

    params = {
        'currency': 'USD',
        'fa_period_type': 'Q',
        'fa_period_offset': bq.func.range('-5Q','0Q'),
        'fa_period_year_end': 'C1231',
        'dates': as_of_date,
        'fa_act_est_data': 'A'
    }

    params_no_currency = {
        'fa_period_type': 'Q',
        'fa_period_offset': bq.func.range('-5Q','0Y'),
        'fa_period_year_end': 'C1231',
        'dates': as_of_date,
        'fa_act_est_data': 'A'
    }

    is_fields = {
        'Revenue': bq.data.sales_rev_turn(**params),
        'Sales and Services Revenues': bq.data.is_s_and_sr_gaap(**params),
        'Financing Revenue': bq.data.is_financing_revenue_gaap(**params),
        'Other Revenue': bq.data.is_other_revenue_gaap(**params),
        'Cost of Revenue': bq.data.is_cogs_to_fe_and_pp_and_g(**params),
        'Cost of Goods & Services Sold': bq.data.is_cog_and_ss_gaap(**params),
        'Cost of Financing Revenue': bq.data.is_cost_of_financing_rev_gaap(**params),
        'Gross Profit': bq.data.gross_profit(**params),
        'Other Operating Income': bq.data.is_other_oper_inc(**params),
        'Operating Expenses': bq.data.is_operating_expn(**params),
        'Selling, General and Administrative Expense': bq.data.is_sg_and_a_expense(**params),
        'R&D Expense Adjusted': bq.data.is_opex_r_and_d_gaap(**params),
        'Other Operating Expenses': bq.data.is_other_operating_expenses_gaap(**params),
        'Operating Income or Losses': bq.data.is_oper_inc(**params),
        'Non-Operating (Income) Loss': bq.data.is_non_operating_inc_loss_gaap(**params),
        'Net Interest Expense': bq.data.is_net_interest_expense(**params),
        'Interest Expense': bq.data.is_int_expense(**params),
        'Interest Income': bq.data.is_int_inc(**params),
        'Foreign Exch Losses (Gains)': bq.data.is_foreign_exch_loss(**params),
        'Other Non-Operating (Income) Loss': bq.data.is_other_nonop_inc_loss_gaap(**params),
        'Pretax Income (Loss), Adjusted': bq.data.pretax_inc(**params),
        'Abnormal Losses (Gains)': bq.data.is_abnormal_item(**params),
        'Merger / Acquisition Expense': bq.data.is_merger_acquisition_expense(**params),
        'Sale of Business': bq.data.is_sale_of_business(**params),
        'Restructuring Expenses': bq.data.is_restructuring_charges(**params),
        'Gain/Loss on Investments': bq.data.is_gain_loss_on_investments(**params),
        'Other Abnormal Items': bq.data.is_other_one_time_items(**params),
        'Pretax Income (Loss), GAAP': bq.data.pretax_inc(**params),
        'Income Tax Expense (Benefit)': bq.data.is_inc_tax_exp(**params),
        'Current Income Tax': bq.data.is_current_income_tax_benefit(**params),
        'Deferred Income Tax': bq.data.is_deferred_income_tax_benefit(**params),
        'Income (Loss) from Continuing Operations': bq.data.is_inc_bef_xo_item(**params),
        'Net Extraordinary Losses (Gains)': bq.data.xo_gl_net_of_tax(**params),
        'Discontinued Operations': bq.data.is_discontinued_operations(**params),
        'Extraordinary Items and Accounting Changes': bq.data.is_extraord_items_and_acctg_chng(**params),
        'Net Income Including Minority Interest': bq.data.ni_including_minority_int_ratio(**params),
        'Net Income/Net Profit (Losses)': bq.data.net_income(**params),
        'Preferred Dividends': bq.data.is_tot_cash_pfd_dvd(**params),
        'Other Adjustments': bq.data.other_adjustments(**params),
        'Net Income Avail to Common, GAAP': bq.data.earn_for_common(**params),
        'Net Income Avail to Common': bq.data.earn_for_common(**params),
        'Net Abnormal Losses (Gains)': bq.data.is_net_abnormal_items(**params),
        'Net Extraordinary Losses (Gains)': bq.data.xo_gl_net_of_tax(**params),
        'Basic Weighted Average Number of Shares': bq.data.is_avg_num_sh_for_eps(**params_no_currency),
        'Basic Earnings per Share': bq.data.is_eps(**params),
        'Basic EPS from Continuing Operations': bq.data.is_earn_bef_xo_items_per_sh(**params),
        'Basic EPS from Continuing Operations': bq.data.is_basic_eps_cont_ops(**params),
        'Diluted Weighted Average Shares': bq.data.is_sh_for_diluted_eps(**params_no_currency),
        'Diluted EPS': bq.data.is_diluted_eps(**params),
        'Diluted EPS from Continuing Operations': bq.data.is_dil_eps_bef_xo(**params),
        'Diluted EPS from Continuing Operations': bq.data.is_dil_eps_cont_ops(**params)
    }
    
    bs_fields = {
        'Cash, Cash Equivalents & STI' : bq.data.c_and_ce_and_sti_detailed(**params),
        'Cash & Cash Equivalents' : bq.data.bs_cash_near_cash_item(**params),
        'ST Investments' : bq.data.bs_mkt_sec_other_st_invest(**params),
        'Accounts & Notes Receiv' : bq.data.bs_acct_note_rcv(**params),
        'Inventories' : bq.data.bs_inventories(**params),
        'Raw Materials' : bq.data.invtry_raw_materials(**params),
        'Work In Process' : bq.data.invtry_in_progress(**params),
        'Finished Goods' : bq.data.invtry_finished_goods(**params),
        'Other Inventory' : bq.data.bs_other_inv(**params),
        'Other ST Assets' : bq.data.other_current_assets_detailed(**params),
        'Derivative & Hedging Assets' : bq.data.bs_deriv_and_hedging_assets_st(**params),
        'Discontinued Operations' : bq.data.bs_assets_of_discontinued_ops_st(**params),
        'Misc ST Assets' : bq.data.bs_other_cur_asset_less_prepay(**params),
        'Total Current Assets' : bq.data.bs_cur_asset_report(**params),
        'Property, Plant & Equip, Net' : bq.data.bs_net_fix_asset(**params),
        'Property, Plant & Equip' : bq.data.bs_gross_fix_asset(**params),
        'Accumulated Depreciation' : bq.data.bs_accum_depr(**params),
        'LT Investments & Receivables' : bq.data.bs_lt_invest(**params),
        'LT Receivables' : bq.data.bs_lt_receivables(**params),
        'Other LT Assets' : bq.data.bs_other_assets_def_chrg_other(**params),
        'Total Intangible Assets' : bq.data.bs_disclosed_intangibles(**params),
        'Goodwill' : bq.data.bs_goodwill(**params),
        'Other Intangible Assets' : bq.data.other_intangible_assets_detailed(**params),
        'Deferred Tax Assets' : bq.data.bs_deferred_tax_assets_lt(**params),
        'Derivative & Hedging Assets' : bq.data.bs_deriv_and_hedging_assets_lt(**params),
        'Prepaid Pension Costs' : bq.data.bs_prepaid_pension_costs_lt(**params),
        'Discontinued Operations' : bq.data.bs_assets_of_discontinued_ops_lt(**params),
        'Misc LT Assets' : bq.data.other_noncurrent_assets_detailed(**params),
        'Total Noncurrent Assets' : bq.data.bs_tot_non_cur_asset(**params),
        'Total Assets' : bq.data.bs_tot_asset(**params),
        'Payables & Accruals' : bq.data.acct_payable_and_accruals_detailed(**params),
        'Accounts Payable' : bq.data.bs_acct_payable(**params),
        'Accrued Taxes' : bq.data.bs_taxes_payable(**params),
        'Interest & Dividends Payable' : bq.data.bs_interest_and_dividends_payable(**params),
        'Other Payables & Accruals' : bq.data.bs_accrual(**params),
        'ST Debt' : bq.data.bs_st_borrow(**params),
        'ST Borrowings' : bq.data.short_term_debt_detailed(**params),
        'ST Finance Leases' : bq.data.st_capital_lease_obligations(**params),
        'ST Operating Leases' : bq.data.bs_st_operating_lease_liabs(**params),
        'Current Portion of LT Debt' : bq.data.bs_curr_portion_lt_debt(**params),
        'Other ST Liabilities' : bq.data.other_current_liabs_sub_detailed(**params),
        'Deferred Revenue' : bq.data.st_deferred_revenue(**params),
        'Derivatives & Hedging' : bq.data.bs_derivative_and_hedging_liabs_st(**params),
        'Discontinued Operations' : bq.data.bs_liabs_of_discontinued_ops_st(**params),
        'Misc ST Liabilities' : bq.data.other_current_liabs_detailed(**params),
        'Total Current Liabilities' : bq.data.bs_cur_liab(**params),
        'LT Debt' : bq.data.bs_lt_borrow(**params),
        'LT Borrowings' : bq.data.long_term_borrowings_detailed(**params),
        'LT Finance Leases' : bq.data.lt_capital_lease_obligations(**params),
        'LT Operating Leases' : bq.data.bs_lt_operating_lease_liabs(**params),
        'Other LT Liabilities' : bq.data.other_noncur_liabs_sub_detailed(**params),
        'Accrued Liabilities' : bq.data.bs_accrued_liabilities(**params),
        'Pension Liabilities' : bq.data.pension_liabilities(**params),
        'Deferred Revenue' : bq.data.lt_deferred_revenue(**params),
        'Derivatives & Hedging' : bq.data.bs_derivative_and_hedging_liabs_lt(**params),
        'Discontinued Operations' : bq.data.bs_liabs_of_discontinued_ops_lt(**params),
        'Misc LT Liabilities' : bq.data.other_noncurrent_liabs_detailed(**params),
        'Total Noncurrent Liabilities' : bq.data.non_cur_liab(**params),
        'Total Liabilities' : bq.data.bs_tot_liab2(**params),
        'Preferred Equity' : bq.data.bs_pfd_eqy(**params),
        'Share Capital & APIC' : bq.data.bs_sh_cap_and_apic(**params),
        'Common Stock' : bq.data.bs_common_stock(**params),
        'Additional Paid in Capital' : bq.data.bs_add_paid_in_cap(**params),
        'Treasury Stock' : bq.data.bs_amt_of_tsy_stock(**params),
        'Retained Earnings' : bq.data.bs_pure_retained_earnings(**params),
        'Other Equity' : bq.data.other_equity_ratio(**params),
        'Equity Before Minority Interest' : bq.data.eqty_bef_minority_int_detailed(**params),
        'Minority/Non Controlling Interest' : bq.data.minority_noncontrolling_interest(**params),
        'Total Equity' : bq.data.total_equity(**params),
        'Total Liabilities & Equity' : bq.data.tot_liab_and_eqy(**params)

    }
    
    price = {
        'Price' : bq.data.px_last(dates=bq.func.range('-12M', as_of_date), currency='USD', fill='prev')
    }
    
    return univ, is_fields, bs_fields, price
