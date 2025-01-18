import boto3
import os
import torch

from transformers import AutoModelForCausalLM


class S3ModelHelper():
    
    # Initialise with the folder for the model
    def __init__(self, s3_sub_folder):
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        
    # move model from local folder to an s3 folder
    def save_model_to_s3(self, local_folder, s3_folder):
        client = boto3.client("s3")
        
        files = os.listdir(local_folder)
        for file in files:
            local_path = f'{local_folder}/{file}'
            obj_name = f'{self.username}/{self.s3_sub_folder}/{s3_folder}/{file}'
            res = client.upload_file(local_path, self.bucket, obj_name)
        print(res)
        
    # need to clear the files from local drive after downloading the model
    def clear_folder(self, local_folder):
        for root, dirs, files in os.walk(local_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
     
    # list all local files in S3
    def list_model_files(self, model_name):
        client = boto3.client("s3")
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}'

        files = []
        for file in client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            files.append(key)
        return files
    
    # re-load the model from s3
    def load_model(self, model_name):
        client = boto3.client("s3")
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}'

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        for file in client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            file_name = model_name + '/' + key[key.find(model_name + '/') + len(model_name) + 1:]
            client.download_file(self.bucket, key, file_name)
        return AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16 )
    
    
    def delete_model_in_s3(self, model_name):
        client = boto3.client("s3")
        folder = f'{self.username}/{self.s3_sub_folder}/{model_name}'
        
        for file in client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            client.delete_object(Bucket=self.bucket, Key=key)
            print(key)
        print("Files deleted in S3")
