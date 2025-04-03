import boto3
import os

class S3Helper:
    def __init__(self, s3_sub_folder):
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.client = boto3.client("s3")

    
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

    