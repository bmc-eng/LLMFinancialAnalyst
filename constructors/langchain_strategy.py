import boto3
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.rate_limiters import InMemoryRateLimiter
from tqdm import tqdm

from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field

import concurrent.futures
from constructors.strategy_construction import StrategyConstruction
from requesters.company_data import SecurityData
from datetime import datetime

class FrontierRun(StrategyConstruction):

    """
    Class to run inference tasks on Frontier models
    in AWS Bedrock
    """
    
    def __init__(self, run_name: str, 
                 model_id: str,  
                 dataset_id: str, 
                 system_prompt:str,
                 boto3_bedrock: boto3.client):
        """
        Constructor method for HuggingfaceRun class
        run_name: str - the name of the backtest run
        run_config: a dictionary containing all of the parameters for the inference backtest
        """
        super().__init__(run_name, 
                         model_id, 
                         dataset_id,
                         system_prompt)

        # set up the system prompt
        self.prompt_template = PromptTemplate.from_template(system_prompt)

        self.company_data = SecurityData('tmp/fs', dataset_id)

        # set up the LLM in Bedrock
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=50,
            check_every_n_seconds=1,
            max_bucket_size=10,
        )

        # set up the LLM
        self.llm = ChatBedrock(
            client = boto3_bedrock,
            model = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            temperature = 0.5,#0.7,
            max_tokens=4000,
            rate_limiter = self.rate_limiter
        )

    def setup_prompts(self):
        """
        Function to set up the prompts per company
        """
        # get a list of the series
        company_dates = self.company_data.date_security_timeseries()
        self.prompts = []
        
        # loop through each of the companies and request the additional information
        for date_security in company_dates:
            date = date_security['date']
            sec = date_security['security']
            security_info = self.company_data.get_financial_data_for_security(date, 
                                                                              sec)
            security_info += self.company_data.get_stock_prices_for_security(date, 
                                                                              sec)
            self.prompts.append({'date': date, 'security':sec, 'info':security_info})
        return self.prompts

    
    def run_model(self, prompt: dict, llm: RunnableSequence) -> dict:
        """
        Function to invoke the LLM on a specific security/ date combination
        """
        # Create the prompt template
        prompt_in = self.prompt_template.format(financials=prompt['info'])
        # invoke the LLM
        output = self.llm.invoke(prompt_in)
        decision_dict = {
            'date': prompt['date'],
            'security': prompt['security'],
            'response': output
        }
        return decision_dict
    
    
    
    def _run_model_specific_backtest(self):
        """
        Function to implement the backtest for LangChain
        """
        self.cached_results = []
        start_time = datetime.now()

        # set up the prompts
        prompts = self.setup_prompts()

         # set up the progress bar
        progress = tqdm(total=len(prompts), position=0, leave=True)

        # Run across 100 workers as executors
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(self.run_model, prompt, self.llm) for prompt in prompts]
            for f in concurrent.futures.as_completed(futures):
                # Update progress
                progress.update(n=1)
                # add the results into the cache
                self.cached_results.append(f.result())
                
        end_time = datetime.now()

        # save the data
        self.save_run(str(end_time-start_time), self.cached_results)

