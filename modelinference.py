import json
import boto3
from s3fs import S3FileSystem
import os
import datetime

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import gather_object

import pandas as pd
from IPython.display import Markdown, display
from ipywidgets import IntProgress, Label, HBox

from helper import get_s3_folder

import company_data
import prompts
from utils.modelHelper import ModelHelper
from utils.logger import Logger

from prompts import SYSTEM_PROMPTS


class RunConfig():
    


class InferenceRun():
    """
    Class to run all inference tasks on one or multi-GPUs
    """
    
    def __init__(self, run_name, run_config):
        self.run_name = run_name # Description of the model run
        self.run_date = datetime.datetime.now()
        
        # model parameters
        self.model_hf_id  = run_config['model_hf_id']   # Model id for tokenizer :str
        self.model_s3_id  = run_config['model_s3_id']   # Load the model from cloud storage :str
        self.model_reload = run_config['model_reload']  # Reload the model from huggingface: bool
        self.model_quant  = run_config['model_quant']   # QuantConfig object from bitsandbytes :QuantConfig
        self.system_prompt= run_config['system_prompt'] # System prompt for inference
        self.multi_gpu    = run_config['multi-gpu']     # Single thread or multi-gpu
        
        # data parameters
        self.dataset      = run_config['dataset']
        self.dataset_loc  = run_config['data_location']
        self.logger       = Logger('tmp/fs')

        

    def load_model_from_hf(self, model_id, useQuantization=False, device='auto'):
        if useQuantization:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=self.model_quant)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
        return model
    
    def load_model_from_storage(model_id_s3, device='auto'):
        helper = ModelHelper('tmp/fs')
        return helper.load_model(model_id_s3, device)
        
        
    def load_model_multi_gpu(self, accelerator):
        if self.model_reload:
            raise Exception("Download the model first to S3 storage before multi-GPU inference!")
        else:
            return load_model_from_storage(self.model_id_s3, device={"":accelerator.process_index})
    
    
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
            model = self.load_model_from_storage(self.model_s3_id)
            
        return model
    
    
    
    def run(self):
        """
        Entry point for running an inference task on a large list of prompts
        This will run in either single model mode or multi-gpu mode
        
        
        """
        if self.multi_gpu:
            accelerator = Accelerator()
            model = self.load_model_multi_gpu(self.model_id_s3, accelerator)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        else:
            model = self.load_model_single()
            print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        

        security_data = company_data.SecurityData('tmp/fs',self.dataset)

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
    