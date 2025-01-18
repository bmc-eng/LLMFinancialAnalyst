import os

# S3 Helper Functionality
user_bucket_name = os.environ['BQUANT_SANDBOX_USER_BUCKET']
bqnt_username = os.environ['BQUANT_USERNAME']

def get_s3_folder():
    return f's3://{user_bucket_name}/{bqnt_username}/tmp/'

