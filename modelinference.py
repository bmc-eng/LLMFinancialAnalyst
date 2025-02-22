import json
import boto3
from s3fs import S3FileSystem
import os
import datetime

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object

import pandas as pd
from IPython.display import Markdown, display
from ipywidgets import IntProgress, Label, HBox

#from helper import get_s3_folder

import company_data
import prompts
import utils.modelHelper as mh
#from utils.modelHelper import ModelHelper
from utils.logger import Logger

from prompts import SYSTEM_PROMPTS


class RunConfig():
    pass


class InferenceRun():
    """
    Class to run all inference tasks on one or multi-GPUs
    """
    
    def __init__(self, run_name, run_config):
        self.run_name = run_name # Description of the model run
        self.run_date = datetime.datetime.now()
        
        # model parameters
        self.model_hf_id  = run_config['model_hf_id']   # Model id for tokenizer :str
        self.model_s3_loc  = run_config['model_s3_loc']   # Load the model from cloud storage :str
        self.model_reload = run_config['model_reload']  # Reload the model from huggingface: bool
        self.model_quant  = run_config['model_quant']   # QuantConfig object from bitsandbytes :QuantConfig
        self.system_prompt= run_config['system_prompt'] # System prompt for inference
        self.multi_gpu    = run_config['multi-gpu']     # Single thread or multi-gpu
        
        # data parameters
        self.dataset      = run_config['dataset']
        self.dataset_loc  = run_config['data_location']
        self.logger       = Logger('tmp/fs')

        

    def load_model_from_hf(self, model_id, useQuantization=False, device='auto'):
        """
        Load the model from Huggingface when a full model refresh is needed
        """
        if useQuantization:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=self.model_quant)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
        return model
    
    def load_model_from_storage(self, model_location, device='auto'):
        """
        Load the model from S3 storage when the model has been pre-saved
        """
        helper = mh.ModelHelper('tmp/fs')
        return helper.load_model(model_location, device)
        
        
    def load_model_multi_gpu(self, accelerator):
        """
        Load the model 
        """
        if self.model_reload:
            raise Exception("Download the model first to S3 storage before multi-GPU inference!")
        else:
            return self.load_model_from_storage(self.model_s3_loc, device={"":accelerator.process_index})
    
    
    def load_model_single(self):
        
        # check if requesting model reload
        if self.model_reload:
            #Reload the model from Huggingface
            if self.model.quant != None:
                
                model = self.load_model_from_hf(self.model_hf_id, True)
            else:
                model = self.load_model_from_hf(self.model_hf_id)
        else:
            # Load model from memory
            print(f'in class: {self.model_s3_loc}')
            model = self.load_model_from_storage(self.model_s3_loc)
            
        return model
    
    
    def create_all_prompts(self):
        """
        Create all of the prompts ready for inference
        """
        company_info = company_data.SecurityData('tmp/fs',self.dataset)
        
        all_prompts = []
        # Get all the dates
        dates = company_info.get_dates()
        # Loop through each date
        for date in dates:
            # Pull out the securities reporting on that date
            securities = company_info.get_securities_reporting_on_date(date)
            # Loop through the securities
            for security in securities:
                # Calculate the prompt
                prompt = company_info.get_prompt(date, security, self.system_prompt)
                record = {'security': security, 'date': date, 'prompt': prompt}
                all_prompts.append(record)
        return all_prompts
    
    def run_model(self, prompt, tokenizer, model):
        tokens = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([tokens], return_tensors='pt').to("cuda")
        generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=5000)
        parsed_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(parsed_ids, skip_special_tokens=True)[0]
    
    
        
    
    def run_multi_gpu(self, log_at=50, start_count=0):
        
        # load the accelerator
        accelerator = Accelerator()
        # load the model
        model = self.load_model_multi_gpu(accelerator)
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)

        
        # Only load the data and calculate all of the prompts once
        if accelerator.is_main_process:
            print("Loading Data")
            # Load and prep the data once
            all_prompts = self.create_all_prompts()
            
            #FOR TESTING ONLY
            all_prompts = all_prompts[:16]
            
            count = start_count
            max_count = len(all_prompts)
            f = IntProgress(min=0, max=max_count) # instantiate the bar
            l = Label(value=str(f.value))
            display(HBox([f,l]))
            
            print("Saving data...")
            # Load all the prompts into local storage
            with open(f'Data/{self.run_name}/prompts.json', 'w') as f:
                json.dump(all_prompts,f)
                
            print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        
        print("waiting...")
        accelerator.wait_for_everyone()
        
        # Load the data back into each GPU memory
        with open(f'Data/{self.run_name}/prompts.json', 'rb') as f:
            all_prompts = json.load(f)
            
        with accelerator.split_between_processes(all_prompts) as prompts:
            results = []
            
            for prompt in prompts:
                response = self.run_model(prompt[prompt], tokenizer, model)
                formatted_response = {'date': prompt['date'], 'security': prompt['security'], 'response': response}
                results.append(formatted_response)
            
            if accelerator.is_main_process:
                count += 1
                f.value += 1
                l.value = str(count) + "/" + str(max_count)
                if count > 0 and count % log_at == 0:
                    results_gathered = gather_object(results)
                    self.logger.log(results_gathered, f"{self.run_name} - {datetime.datetime.now()}.json")
        
        if accelerator.is_main_process:
            results_gathered = gather_object(results)
            self.logger.log(results_gathered, f"results - {self.run_name}")
        
        

    
    
    def run(self):
        """
        Entry point for running an inference task on a large list of prompts
        This will run in either single model mode or multi-gpu mode
        
        """
        start_time = datetime.datetime.now()
        
        if self.multi_gpu:
            self.run_multi_gpu()
        else:
            model = self.load_model_single()
            print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        

        

        # set up system prompts
        all_prompts = create_all_prompts(company_data, system_prompt)

        # batch into groups of 8
        #batches = [all_prompts[i:i + 8] for i in range(0, len(all_prompts), 8)]  

        accelerator.wait_for_everyone()
        # Limit for testing
        #prompt_limit = all_prompts[:5]

    
        #for batch in batches:
            #run the backtest
        run_backtest(all_prompts, tokenizer, model, self.logger, accelerator)
    