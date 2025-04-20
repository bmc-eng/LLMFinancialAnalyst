import os
import json
import boto3
from s3fs import S3FileSystem

import pandas as pd

# Helper class to manage the quarterly point in time datasets
class SecurityData:
    """Object to retrieve all of the downloaded company data from S3.
    dataset_folder - Folder in S3 with all of the data
    dataset_name - name of the dataset
    use_local - dataset can be passed to the object to speed up reloading"""
    # Initiatise with a dataset. 
    def __init__(self, dataset_folder, dataset_name, use_local=None):
        self.data = {}
        
        if use_local == None:
            try:
                user_bucket_name = os.environ['BQUANT_SANDBOX_USER_BUCKET']
                bqnt_username = os.environ['BQUANT_USERNAME']
    
                path_to_s3 = f's3://{user_bucket_name}/{bqnt_username}/{dataset_folder}/{dataset_name}'
                s3 = S3FileSystem()
    
                with s3.open(path_to_s3, 'rb') as f:
                    self.data = json.load(f)
            except:
                raise Exception("Company data must be downloaded first before it is requested.")
        else:
            self.data = use_local
    
    def get_unique_securities(self):
        """Function to get all of the unique securities in a dataset"""
        secs = []
        for date in self.data.keys():
            for sec in self.data[date].keys():
                if sec not in secs:
                    secs.append(sec)
        return secs
    
    # Get all the dates for backtesting
    def get_dates(self):
        return list(self.data.keys())
    
    # Get all the securities reporting on a given date
    def get_securities_reporting_on_date(self, date):
        try:
            return list(self.data[date].keys())
        except:
            return 'no data'
    
    # Get the prompt for a security on a given date with all three statements
    def get_prompt(self, date, security, system_prompt):
        is_statement = self.get_security_statement(date, security, 'is')
        bs_statement = self.get_security_statement(date, security, 'bs')
        px_values = self.get_security_statement(date, security, 'px')
        
        company_info = "Income Statement:" + is_statement.to_string() + "\n Balance Sheet: " + bs_statement.to_string() + "\n Historical Price: " + px_values.to_string()
        
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": company_info},

        ]
        return prompt
    
    def total_securities_in_backtest(self):
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
        
    
    
    def get_all_data(self):
        return self.data
    
