import os
import json
import boto3
from s3fs import S3FileSystem
from utils.s3_helper import S3Helper

import pandas as pd

class SecurityData:
    """
    Class to retrieve all of the downloaded company data from S3.
    dataset_folder:      str - Folder in S3 with all of the data
    dataset_name:        str - name of the dataset
    use_local (Optional) bool - dataset can be passed to the object to speed up reloading
    """
    # Initiatise with a dataset. 
    def __init__(self, dataset_folder, dataset_name, use_local=None):
        self.data = {}
        
        if use_local == None:
            try:
                s3h = S3Helper(dataset_folder)
                s3h.get_file(dataset_name, f'/tmp/{dataset_name}')
    
                with open(f'/tmp/{dataset_name}', 'rb') as f:
                    self.data = json.load(f)
            except:
                raise Exception("Company data must be downloaded first before it is requested.")
        else:
            self.data = use_local

    
    def get_unique_securities(self) -> list[str]:
        """
        Function to get all of the unique securities in a dataset
        Return: List of securities
        """
        secs = []
        for date in self.data.keys():
            for sec in self.data[date].keys():
                if sec not in secs:
                    secs.append(sec)
        return secs
    
    
    def get_dates(self) -> list[str]:
        """
        Function to get all of the dates in the dataset for backtesting
        Return: List of dates
        """
        return list(self.data.keys())
    
    
    def get_securities_reporting_on_date(self, date:str) -> list[str]:
        """
        Function to get all of the securities reporting on a given date
        Return: List of securities
        """
        try:
            return list(self.data[date].keys())
        except:
            return 'no data'
    

    def get_security_all_data(self, date, security) -> dict:
        sec_fs = self.get_financial_data_for_security(date, security)
        stock_price = self.get_stock_prices_for_security(date,security)
        name = self.all_data[date][security]['mt']['name']
        figi = self.all_data[date][security]['mt']['figi']
        sector = self.all_data[date][security]['mt']['sector']
        return name, figi, sector, sec_fs, stock_price
    

    def get_stock_prices_for_security(self, date, security):
        px_values = self.get_security_statement(date, security, 'px')
        return "\n Historical Price: " + px_values.to_string()
    
    
    def get_financial_data_for_security(self, date, security):
        is_statement = self.get_security_statement(date, security, 'is')
        bs_statement = self.get_security_statement(date, security, 'bs')
        

        company_info = "Income Statement:" + is_statement.to_string() + "\n Balance Sheet: " + bs_statement.to_string() 

        return company_info
        
    
    def get_prompt(self, date, security, system_prompt) -> str:
        """
        Construct a prompt for a security on a date in a chat format
        date:          str - Date of rebalance
        security:      str - Security reporting on given date
        system_prompt: str - The system prompt to use in construction of the prompt
        Return: prompt string
        """
        is_statement = self.get_security_statement(date, security, 'is')
        bs_statement = self.get_security_statement(date, security, 'bs')
        px_values = self.get_security_statement(date, security, 'px')
        
        company_info = "Income Statement:" + is_statement.to_string() + "\n Balance Sheet: " + bs_statement.to_string() + "\n Historical Price: " + px_values.to_string()
        
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_info},
        ]
        return prompt


    def date_security_timeseries(self):
        """
        Returns all of the dates and securities in order of time.
        """
        output_list = []
        dates = self.get_dates()
        for date in dates:
            securities_reporting = self.get_securities_reporting_on_date(date)

            for security in securities_reporting:
                output_list.append({'date': date, 'security': security})

        return output_list
        
    
    
    
    def total_number_securities(self) -> int:
        """
        Function to return the total number of securities
        """
        count = 0
        for date in self.get_dates():
            count += len(self.get_securities_reporting_on_date(date))
        return count

        
    
    def get_security_statement(self, date, security, statement_type):
        # Statement type must be of 'is', 'bs' or 'px'
        try:
            # Load the security/ date/ statement and convert to DataFrame
            statement_data = json.loads(self.data[date][security][statement_type])
            df_sec = pd.DataFrame(json.loads(statement_data))
            
            # Fix issue with the date format appearing incorrectly in JSON
            if statement_type == 'px':
                df_fix = df_sec.reset_index()
                df_fix['Date'] = pd.to_datetime(df_fix['index'], unit='ms')
                return df_fix.set_index('Date')[['Price']]
            else:
                return self._clean_security_info(df_sec)
        except Exception as err:
            return err
    
    # Clean up the dataframe to reduce the number of tokens
    def _clean_security_info(self, df):
        # drop the numbers from the labels and remove Adj
        df['items'] = [s[3:].replace(' (Adj)','') for s in df.index]
        df = df.set_index('items')
        return df.drop_duplicates()
        
    
    
    def get_all_data(self) -> dict:
        """
        Function to return all datasets 
        """
        return self.data
    
