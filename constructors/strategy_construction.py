import json
import os

from requesters.company_data import SecurityData
from utils.s3_helper import S3Helper
from datetime import datetime

from typing import Dict, TypedDict, Optional, Annotated


class StrategyConstruction():
    """
    Strategy Construction Base Class. This is used by the inference run classes. 
    """

    def __init__(self, run_name:str, 
                 model_id: str,  
                 dataset_id: str, 
                 system_prompt:str):
        """
        Constructor method for the StrategyConstruction class
        run_name: str - the name of the backtest run
        run_config: a dictionary containing all of the parameters for the inference backtest
        """
        self.run_name: str = run_name # Description of the model run
        self.run_date: str = datetime.now()
        self.model_id: str = model_id
        self.dataset_id: str = dataset_id
        self.system_prompt: str = system_prompt

        # Set up the helpers
        self.company_info = SecurityData('tmp/fs',self.dataset_id)
        self.s3h = S3Helper('tmp/fs')

        # Global variable to cache results. - This can still be accessed if the backtest fails.
        self.cached_results = []

        # Store in the temporary store of Bloomberg Lab
        self.project_folder = f'/tmp/{run_name}'
        if not os.path.exists(self.project_folder):
            os.makedirs(self.project_folder)


    def run(self, **kwargs):
        """
        Function to run the backtest
        """
        # record the start time
        start_time = datetime.now()

        # call the backtest function
        self._run_model_specific_backtest()

        # record the end time
        end_time = datatime.now()

        # save the results
        self.save_run(str(end_time - start_time))

    def _run_model_specific_backtest(self):
        """
        Overwrite function - specific to each model type
        """
        raise NotImplementedError("Base class - should not be called directly!")

    def run_model(self, **kwargs):
        """
        Overwrite function - specific to each model type
        """
        raise NotImplementedError("Base class - should not be called directly!")
    
    def cache_results(self, security: str, date: str, **kwargs):
        self.cached_results.append({
            'security': security,
            'date': date,
            **kwargs
        })
    
    def save_run(self, time_taken:str, results):
        """
        Function to store the results of a strategy locally and in Bloomberg Lab S3 Storage
        time_take: str - time taken to run the entire strategy run.
        """
        self.cached_results = results
        self.output = {
            'run_name': str(self.run_name),
            'run_date': str(self.run_date), 
            'system_prompt': self.system_prompt, 
            'dataset': self.dataset_id, 
            'model': self.model_id, 
            'results': results
        }
        try:
            local_filename = f'Results/{self.run_date}.json'
            with open(local_filename, 'w') as f:
                json.dump(self.output, f)
            self.s3h.add_file(local_filename=local_filename)
        except:
            print("unable to save file - you can access via self.output to cache your results from the backtest!")
        
        

    
    