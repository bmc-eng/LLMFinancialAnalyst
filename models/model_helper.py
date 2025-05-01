import boto3
import os
import torch
import json
import transformers

from utils.s3_helper import S3Helper
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from s3fs import S3FileSystem



class ModelHelper(S3Helper):

    def __init__(self, project_folder: str):
        """
        Constructor for ModelHelper. This class is to help manage the saving and 
        loading of Huggingface models into BQuant Enterprise S3 storage.
        project_folder: str - Project files in BQuant Enterprise S3 storage
        """
        super().__init__(project_folder)

    
    def load_model_from_hf(self, model_id:str, 
                           use_quantization:bool=False, 
                           quant_config:BitsAndBytesConfig=None, 
                           device='auto') -> AutoModelForCausalLM:
        """
        Load the model from Huggingface when a full model refresh is needed. 
        Requires an active login with Huggingface
        Returns a Huggingface AutoModelForCausalLM model
        """
        try:
            if use_quantization:
                model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                             device_map=device, 
                                                             quantization_config=quant_config)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                             device_map=device, 
                                                             torch_dtype=torch.bfloat16)
            return model
        except:
            raise Exception("Error in loading model from Huggingface. Check login status and model id!")
            
    
    def save_model_locally(self, model, model_name, model_folder):
        """
        Save the model in a local file directory in the project
        """
        local_filepath = f'{model_folder}/{model_name}'
        model.save_pretrained(local_filepath)

    
    def get_model_and_save(self, 
                           model_id, 
                           model_name,
                           model_tmp_location,
                           use_quantization=False, 
                           quant_config=None, 
                           device='auto'):
        model = self.load_model_from_hf(model_id, use_quantization, quant_config, device)
        # save file locally first
        self.save_model_locally(model, model_name, model_tmp_location)
        #push to the cloud
        self.save_model_to_s3(f'{model_tmp_location}/{model_name}', model_name)
        # remove the local folder
        self.clear_folder(f'{model_tmp_location}/{model_name}')
    
    
    
    def save_model_to_s3(self,local_model_folder: str, s3_model_folder: str):
        """
        Save a Huggingface model to BQuant Enterprise S3 Storage
        local_model_folder: str - location in the project folder of the model
        s3_model_name: str - location in S3 to store the model
        """
        super().add_folder(local_model_folder, s3_model_folder)
        
    
    def clear_folder(self, local_folder, count=0):
        """
        Function to delete all files and folders in a directory.
        local_folder: str - the local file directory in the project folder to clear
        
        """
        super().clear_local_folder(local_folder, count)

    
    def list_model_files(self, model_name) -> list[str]:
        """
        List all of the files for a particular model in BQuant Enterprise S3
        model_name: str - Name of the model to find
        """
        return super().list_folder(model_name)
    
    
    def load_model(self, model_name: str, device: str = 'auto', remove_local_once_loaded:bool = False) -> AutoModelForCausalLM:
        """
        Load the model from an S3 bucket
        model_name: Location in S3 of the model within the subfolder
        device:     Returns the model with either 'auto' or accelerate device for multi-gpu inference
        returns:    Huggingface Model
        """
        super().get_folder(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16 )

        if remove_local_once_loaded:
            self.clear_folder(model_name)

        return model
        
    
    
    def delete_model(self, model_name):
        """
        Delete the model from the S3 bucket
        model_name: str - Model to delete from BQuant Enterprise S3 storage
        """
        super().delete_folder(model_name)
        print(f'Model deleted: {model_name}')