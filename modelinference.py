import json
import boto3
from s3fs import S3FileSystem
import os
import datetime

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import gather_object

import pandas as pd
from IPython.display import Markdown, display
from ipywidgets import IntProgress, Label, HBox

#from helper import get_s3_folder
import company_data
import prompts
import utils.model_helper as mh
from utils.logger import Logger

from tqdm import tqdm


class InferenceRun():
    """
    Class to run all inference tasks on one or multi-GPUs
    """
    
    def __init__(self, run_name, run_config):
        self.run_name: str = run_name # Description of the model run
        self.run_date: str = datetime.datetime.now()
        self.helper: mh.ModelHelper = mh.ModelHelper('tmp/fs')
        
        # model parameters
        self.model_hf_id: str                 = run_config['model_hf_id']   # Model id for tokenizer :str
        self.model_s3_loc: str                = run_config['model_s3_loc']   # Load the model from cloud storage :str
        self.model_reload: bool               = run_config['model_reload']  # Reload the model from huggingface: bool
        self.model_quant: BitsAndBytesConfig  = run_config['model_quant']   # QuantConfig object from bitsandbytes :QuantConfig
        self.system_prompt: str               = run_config['system_prompt'] # System prompt for inference
        self.multi_gpu: bool                  = run_config['multi-gpu']     # Single thread or multi-gpu
        
        # data parameters
        self.dataset      = run_config['dataset']
        self.dataset_loc  = run_config['data_location']
        self.logger       = Logger('tmp/fs')
        
        self.project_folder = f'Data/{run_name}'
        if not os.path.exists(self.project_folder):
            os.makedirs(self.project_folder)

        

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
        return self.helper.load_model(model_location, device)
        
        
    def load_model_multi_gpu(self, accelerator):
        """
        Load the model 
        """
        if self.model_reload:
            raise Exception("Download the model first to S3 storage before multi-GPU inference!")
        else:
            return self.load_model_from_storage(self.model_s3_loc, device={"":accelerator.process_index})
    
    
    def load_model_single(self):
        """
        Load the model into a single GPU
        """
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
    
    
    def save_run(self, results):
        """
        Save the results of the inference run at the end
        results:     List of JSON objects from each model inference task
        """
        print("Called Save run")
        
        with open(f'{self.project_folder}/results - {self.run_date}.json', 'w') as f:
            json.dump(results, f)
            
        self.logger.log(results, f"Results_{self.run_date}.json")
        print("Run Completed!")
        
    
    
    def create_all_prompts(self, is_save_prompts=False):
        """
        Create all of the prompts ready for inference
        """
        if not os.path.exists(f'Data/{self.run_name}/prompts.json'):
            print("Requesting all datasets...")
            company_info = company_data.SecurityData('tmp/fs',self.dataset_loc)

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
                    prompt = company_info.get_prompt(date, security, self.system_prompt['prompt'])
                    record = {'security': security, 'date': date, 'prompt': prompt}
                    all_prompts.append(record)

            if is_save_prompts:
                print("Saving data...")
                # Load all the prompts into local storage
                with open(f'{self.project_folder}/prompts.json', 'w') as f:
                    json.dump(all_prompts,f)
            return all_prompts
        else:
            with open(f'Data/{self.run_name}/prompts.json') as f:
                all_prompts = json.load(f)
        
            return all_prompts
    
    
    def run_model(self, prompt, tokenizer, model):
        """
        Perform a single inference run with a prompt, tokenizer and model
        prompt:       string of the prompt to run through the model
        tokenizer:    the tokenizer to encode and decode tokens into the model
        model:        huggingface model
        returns       string decoded inference output of the model
        """
        tokens = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True )
        model_inputs = tokenizer([tokens], return_tensors='pt').to("cuda")
        generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens=2000)
        parsed_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(parsed_ids, skip_special_tokens=True)[0]
    
    
    def format_json(self, llm_output):
        # remove all the broken lines
        form = llm_output.replace('\n','')
        # Find the start and end of the JSON input
        try:
            soj = form.find('```json')
            eoj = form.find('}```')
            
            if eoj == -1:
                eoj = len(llm_output)
                llm_output = llm_output + '}```'
            # Pull out the additional context
            additional = form[:soj]
            additional += form[eoj + 4:]
            json_obj = json.loads(form[soj + 7:eoj + 1])
            json_obj['AdditionalContext'] = additional
            return json_obj
        except:
            return llm_output
        
    
    def run_multi_gpu(self, log_at: int=1, start_count: int=0):
        """Run an inference task with multiple GPUs. This is the entry point for a multi-gpu task"""
        
        # load the accelerator
        accelerator = Accelerator()
        # load the model
        model = self.load_model_multi_gpu(accelerator) 
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)

        
        # Only load the data and calculate all of the prompts once
        if accelerator.is_main_process:
            
            start_time = datetime.datetime.now()
            # Load and prep the data once
            all_prompts = self.create_all_prompts(True)
            progress = tqdm(total=len(all_prompts), position=0, leave=True)
            count = start_count
              
            print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        
        print("Waiting...")
        accelerator.wait_for_everyone()
        
        # Load the data back into each GPU memory
        with open(f'{self.project_folder}/prompts.json', 'rb') as f:
            all_prompts = json.load(f)
            
        #FOR TESTING ONLY
        #all_prompts = all_prompts[:8]
        
        # Clear the memory to free up space in local disk
        self.helper.clear_folder(self.model_s3_loc)

        all_prompts = all_prompts[800:]
            
        with accelerator.split_between_processes(all_prompts) as prompts:
            results = []
            print("starting backtest...")
            
            for prompt in prompts:
                try:
                    response = self.run_model(prompt['prompt'], tokenizer, model)
                    formatted_response = {'date': prompt['date'], 'security': prompt['security'], 'response': self.format_json(response)}
                    results.append(formatted_response)
                
                    if accelerator.is_main_process:
    
                        # Update progress
                        count += 1
                        progress.update(accelerator.num_processes)
                except Exception as e:
                    print(f"Process {torch.multiprocessing.current_process().name} crashed: {e}")
                

                    # if count > 0 and count % log_at == 0:
                    #     #results_gathered = gather_object(results)
                    #     print("gathered results")
                    #     self.logger.log(results_gathered, f"{self.run_name} - {datetime.datetime.now()}.json")
        
        print("Finished run...")
        accelerator.wait_for_everyone()
        results_gathered = gather_object(results)
        accelerator.wait_for_everyone()
        print("Gathered results...")
        
        if accelerator.is_main_process:
            end_time = datetime.datetime.now()
            print(f"Finished run in {end_time - start_time}")
            end_result = {'run_date': str(self.run_date), 'system_prompt': self.system_prompt['prompt'], 'dataset': self.dataset, 'model': self.model_hf_id, 'results': results_gathered}
            self.save_run(end_result)
            
        accelerator.wait_for_everyone()
        

    def run_single(self, log_at: int=1, start_count: int=0):
        
        # load the model
        model = self.load_model_single() 
        tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)

        # Start the timer      
        start_time = datetime.datetime.now()
        
        # Load and prep the data once
        all_prompts = self.create_all_prompts(True)
        
        # test run
        all_prompts = all_prompts[:4]
        progress = tqdm(total=len(all_prompts), position=0, leave=True)
        count = start_count
              
        print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        
        # Clear the memory to free up space in local disk
        self.helper.clear_folder(self.model_s3_loc)
            
        
        results = []
        print("starting backtest...")
            
        for prompt in all_prompts:
            response = self.run_model(prompt['prompt'], tokenizer, model)
            formatted_response = {'date': prompt['date'], 'security': prompt['security'], 'response': self.format_json(response)}
            results.append(formatted_response)

            # Update progress
            count += 1
            progress.update(count)
        
        end_time = datetime.datetime.now()
        print(f"Finished run in {end_time - start_time}")
        end_result = {'run_date': str(self.run_date), 'system_prompt': self.system_prompt['prompt'], 'dataset': self.dataset, 'model': self.model_hf_id, 'results': results_gathered}
        self.save_run(end_result)
    
    
    def run(self):
        """
        Entry point for running an inference task on a large list of prompts
        This will run in either single model mode or multi-gpu mode
        
        """
        
        
        if self.multi_gpu:
            self.run_multi_gpu()
        else:
            model = self.load_model_single()
            print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
