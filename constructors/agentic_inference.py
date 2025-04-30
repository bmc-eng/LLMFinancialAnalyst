import copy
import pandas as pd
import requesters.company_data as cd
import agents.financial_agent as fa
import agents.committee_agent as ca
import importlib
from tqdm import tqdm
import json

from utils.s3_helper import S3Helper
from datetime import datetime

from strategy_construction import StrategyConstruction

import concurrent.futures


class AgenticRun(StrategyConstruction):
    """
    Implementation of the Agentic Strategy Construction class
    """

    def __init__(self, run_name:str, 
                 model_id: str,  
                 dataset_id: str, 
                 system_prompt:str):
        super().__init__(run_name, model_id, dataset_id, system_prompt)
        
        # Get the news datasets needed for the model to run
        self.s3h.get_file(filename='dow_headlines.parquet', local_filename='/tmp/dow_headlines.parquet')
        self.news_headlines = pd.read_parquet('/tmp/dow_headlines.parquet')

        self.financial_agent = fa.FinancialAnalystAgent()
        self.committee_agent = ca.CommitteeAgent()


    
    def run_model(self, security: str, as_of_date: str) -> dict:
        """
        Function to run a single run of the Agent
        """
        company_data = self.company_info.get_security_all_data(as_of_date, security)
        # Time the run
        start_time = datetime.now()
        # Run the financial analyst agent
        financial_report = self.financial_agent.run(security_data=company_data, 
                                           news_data=news_headlines, 
                                           as_of_date=as_of_date)
        # Run the committee agent
        committee_report = self.committee_agent.run(senior_analyst_report=financial_report['senior_report'],
                                               financial_statement_analysis=financial_report['financial_report'],
                                                    security_data=company_data)
        # record the end time
        end_time = datetime.now()

        # output into a dictionary to store the results
        decision_dict = {
            'date': as_of_date,
            'security': security,
            'earning_decision': financial_report['final_output'].direction,
            'earning_magnitude': financial_report['final_output'].magnitude,
            'earning_confidence': financial_report['final_output'].confidence,
            'recommendation': committee_report['results'].recommendation,
            
            'responses': {'financial_analyst': financial_report,
                         'committee_report': committee_report},
            'time': str(end_time - start_time)
        }
        return decision_dict

    
    def _run_model_specific_backtest(self):

        # Get the dataset prompts
        dates_and_securities = self.company_info.date_security_timeseries()

        # set up the progress bar
        progress = tqdm(total=len(dates_and_securities), position=0, leave=True)

        # set up multi-threaded workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = [executor.submit(self.run_model, single['security'], single['date']) for single in dates_and_securities]
            for f in concurrent.futures.as_completed(futures):
                # Update progress
                progress.update(n=1)
                # add the results into the cache
                self.cached_results.append(f.result())


    def save_run(self, time_taken:str):


        # Specific to the Agentic class to convert LangChain objects to string before saving.
        for item in self.cached_results:
            item['responses']['financial_analyst']['final_output'] = str(item['responses']['financial_analyst']['final_output'])
            item['responses']['committee_report']['results'] = str(item['responses']['committee_report']['results'])
            item['responses']['committee_report']['history'] = str(item['responses']['committee_report']['history'])
            item['time']=str(item['time'])

        # Define the backtest object
        self.output = {
            'run_name': str(self.run_name),
            'run_date': str(self.run_date), 
            'system_prompt': self.system_prompt, 
            'dataset': self.dataset_id, 
            'model': self.model_hf_id, 
            'results': self.cached_results
        }
        try:
            local_filename = f'/tmp/{run_name} - {self.run_date}.json'
            with open(local_filename, 'w') as f:
                json.dump(self.output, f)
            s3h.add_file(local_filename=local_filename)
        except:
            print("unable to save file - you can access via self.output to cache your results from the backtest!")


    def create_trade_report(self, trade_only=False) -> pd.DataFrame:
        """
        Function to create the trade report to pass into the Strategy Analysis tool
        """
        trade_report = []
        if trade_only:
            for item in self.cached_data:
                trade_report.append({'date': item['date'], 
                                     'security': item['security'], 
                                     'decision': item['recommendation'],
                                     'confidence': item['earning_confidence']})
        
            return pd.DataFrame(data=trade_report)
        else:
            for item in self.cached_data:
                trade_report.append({'date': item['date'], 
                                     'security': item['security'], 
                                     'decision': item['recommendation'],
                                     'confidence': item['earning_confidence'],
                                     'earning_decision': item['earning_decision'],
                                     'earning_magnitude': item['earning_magnitude']})
            return trade_report






        
        