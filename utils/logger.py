import boto3
import os
import torch
import json


class Logger:
    
    """
    Class to log output from the model inference runs or training back to S3
    """
    
    def __init__(self, s3_sub_folder):
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.s3 = S3FileSystem()
        self.client = boto3.client("s3")
        
    def _write(self, data, filepath):
        
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