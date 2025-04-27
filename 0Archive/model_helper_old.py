import boto3
import os
import torch
import json
import transformers

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from s3fs import S3FileSystem


class ModelHelper():
    """Class to help manage the saving and loading of Huggingface models into S3 storage."""
    
    # Initialise with the folder for the model
    def __init__(self, s3_sub_folder: str):
        """Constructor: Initialise with a subfolder in the Bloomberg Lab S3 bucket.
        s3_sub_folder: str a sub folder to save models in S3"""
        
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.client = boto3.client("s3")
        
    def _get_model(self, model_name: str):
        """Internal function to get a model from S3 by the model name"""
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}/'

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            file_name = model_name + '/' + key[key.find(model_name + '/') + len(model_name) + 1:]
            self.client.download_file(self.bucket, key, file_name)
    

    def load_model_from_hf(self, model_id:str, 
                           use_quantization:bool=False, 
                           quant_config:BitsAndBytesConfig=None, 
                           device='auto') -> AutoModelForCausalLM:
        """
        Load the model from Huggingface when a full model refresh is needed. Returns a model
        """
        if use_quantization:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=quant_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
        return model

    def save_model_locally(self, model, model_name, model_folder):
        local_filepath = f'{model_folder}/{model_name}'
        model.save_pretrained(local_filepath)
    
    def get_model_and_save(self, model_id, 
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
    
    
    # move model from local folder to an s3 folder
    def save_model_to_s3(self, local_folder, s3_folder):
        
        files = os.listdir(local_folder)
        for file in files:
            local_path = f'{local_folder}/{file}'
            obj_name = f'{self.username}/{self.s3_sub_folder}/{s3_folder}/{file}'
            res = self.client.upload_file(local_path, self.bucket, obj_name)
        print(res)
        

    def list_all_folders(self):
        """Function to list all folders and files in the sub folder"""
        files = []
        for file in self.client.list_objects(Bucket=self.bucket, Prefix=f'{self.username}/{self.s3_sub_folder}')['Contents']:
            key = file['Key']
            print(key)
    
    
    def clear_folder(self, local_folder, count=0):
        """Function to delete all files and folders in a directory."""
        try:
            for root, dirs, files in os.walk(local_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except: 
            # check not stuck in loop - needed for multi-threading tasks
            if count <= 3:
                self.clear_folder(local_folder, count + 1)
     
    # list all local files in S3
    def list_model_files(self, model_name):
    
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}'

        files = []
        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            files.append(key)
        return files
    
    # re-load the model from s3
    def load_model(self, model_name, device):
        """
        Load the model from an S3 bucket
        model_name: Location in S3 of the model within the subfolder
        device:     Returns the model with either 'auto' or accelerate device for multi-gpu inference
        """
        print(model_name)
        self._get_model(model_name)
        
        return AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16 )
        
    
    
    def delete_model_in_s3(self, model_name):
        """
        Delete the model from the S3 bucket
        """
        client = boto3.client("s3")
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}'
        
        for file in client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            client.delete_object(Bucket=self.bucket, Key=key)
            print(key)
        print("Files deleted in S3")
        