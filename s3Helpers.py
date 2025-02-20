import boto3
import os
import torch
import json

from transformers import AutoModelForCausalLM
from s3fs import S3FileSystem

# Class to help with loading models
class S3ModelHelper():
    
    # Initialise with the folder for the model
    def __init__(self, s3_sub_folder):
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.client = boto3.client("s3")
        
    def _get_model(self, model_name):
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}/'

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            file_name = model_name + '/' + key[key.find(model_name + '/') + len(model_name) + 1:]
            self.client.download_file(self.bucket, key, file_name)
    
    
    # move model from local folder to an s3 folder
    def save_model_to_s3(self, local_folder, s3_folder):
        
        files = os.listdir(local_folder)
        for file in files:
            local_path = f'{local_folder}/{file}'
            obj_name = f'{self.username}/{self.s3_sub_folder}/{s3_folder}/{file}'
            res = self.client.upload_file(local_path, self.bucket, obj_name)
        print(res)
        
    # need to clear the files from local drive after downloading the model
    def clear_folder(self, local_folder, count=0):
        try:
            for root, dirs, files in os.walk(local_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except: 
            # check not stuck in loop
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
    def load_model(self, model_name, accelerator=None):
        self._get_model(model_name)
        
        if accelerator == None:
            return AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16 )
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, device_map={"":accelerator.process_index}, torch_dtype=torch.bfloat16)
    
    
    def delete_model_in_s3(self, model_name):
        client = boto3.client("s3")
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}'
        
        for file in client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            client.delete_object(Bucket=self.bucket, Key=key)
            print(key)
        print("Files deleted in S3")

class Logger:
    def __init__(self, s3_sub_folder):
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.s3 = S3FileSystem()
        self.client = boto3.client("s3")
        
    def _write(self, data, filepath):
        print(filepath)
        with self.s3.open(filepath, 'w') as file:
            json.dump(data, file)
            
        print("Saved " + filepath)
    
    def log(self, data, filename):
        
        path_to_s3 = f's3://{self.bucket}/{self.username}/{self.s3_sub_folder}/logs/{filename}'
        self._write(data, path_to_s3)
        
        
    def get_list_of_logs(self):
        folder = f'{self.username}/{self.s3_sub_folder}/logs'

        files = []
        try:
            for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
                key = file['Key']
                files.append(key)
        except KeyError:
            pass
        return files
    
    def get_log(self, logname):
        path_to_file = f's3://{self.bucket}/{self.username}/{self.s3_sub_folder}/logs/{logname}'
        
        with self.s3.open(path_to_file, 'rb') as file:
            data = json.load(file)
            
        return data
    
    def create_master_log(self, save_to_s3=True, filename='output_logs.json'):
                
        log_list = self.get_list_of_logs()
        logs = []
        for logfile in log_list:
            logs += self.get_log(logfile[logfile.find('/logs/') + 6:])
            
        if save_to_s3:
            path_to_s3 = f's3://{self.bucket}/{self.username}/{self.s3_sub_folder}/{filename}'
            data = {'content': logs}
            self._write(data, path_to_s3)
            
        return logs
    
    def clear_all_logs(self):

        folder = f'{self.username}/{self.s3_sub_folder}/logs'
        
        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            self.client.delete_object(Bucket=self.bucket, Key=key)
            print("Deleted: ", key)
        print("Files deleted in S3")
        
        
   
        
    
    