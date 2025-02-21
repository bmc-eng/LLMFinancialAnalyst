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
    
    def load_model_from_storage(model_id, device='auto'):
        helper = ModelHelper('tmp/fs')
        model = helper.load_model(model_id_s3, device)
        
        
    
    def load_model(self):
        
        # check if requesting model reload
        if self.model_reload:
            if USE_QUANTIZATION:
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":accelerator.process_index}, quantization_config=quant_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":accelerator.process_index}, torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            # load the pre-saved model from S3
            model_helper = s3Helpers.S3ModelHelper(s3_sub_folder='tmp/fs')
            model = model_helper.load_model(model_id_s3, accelerator)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            #model_helper.clear_folder(model_id_s3)

        print(f"Memory footprint: {model.get_memory_footprint() / 1e9:,.1f} GB")
        return model, tokenizer
    
    
    
    def run(self):
        """
        Entry point for running an inference task on a large list of prompts
        This will run with multi-gpus when called
        
        
        """
        if self.multi_gpu:
            accelerator = Accelerator()
            model, tokenizer = load_model(self.model_id, self.model_id_s3, accelerator)

            accelerator.wait_for_everyone()


        

        company_data = get_all_data()

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
    