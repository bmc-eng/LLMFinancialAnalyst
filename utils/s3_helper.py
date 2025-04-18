import boto3
import os

class S3Helper:
    """Helper class to handle files in a folder moving to S3 storage
    in the Bloomberg Lab sandbox. Initialize with a project folder name for S3
    s3_sub_folder: str - sub folder in Bloomberg Lab to save project data"""
    def __init__(self, s3_sub_folder: str):
        self.username = os.environ['BQUANT_USERNAME']
        self.s3_sub_folder = s3_sub_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.client = boto3.client("s3")

    def _list_local_files(self, local_folder:str) -> list[str]:
        """Internal function to list all of the files in the local folder"""
        file_directory = []
        for root, dirs, files in os.walk(local_folder, topdown=False):
            for name in files:
                
                file_directory.append(os.path.join(root, name))
            for name in dirs:
                pass
        return file_directory
    
    
    def add_folder(self, local_folder: str, s3_folder: str):
        """upload all files in a local folder in a folder to S3"""
        # Walk through each of the files in the foldr structure
        files = self._list_local_files(local_folder)
        for file in files:
            # remove the local folder name
            obj_name = f'{self.username}/{self.s3_sub_folder}/{s3_folder}{file[len(local_folder):]}'
            res = self.client.upload_file(file, self.bucket, obj_name)
        print('Uploaded')

    
    def list_folder(self, s3_folder:str = None) -> list[str]:
        """list all of the files in an s3 folder.
        s3_folder: str - the sub folder in Bloomberg Lab S3 storage
        Return: List of files"""
        if s3_folder != None:
            folder = f'{self.username}/{self.s3_sub_folder}/{s3_folder}'
        else:
            folder = f'{self.username}/{self.s3_sub_folder}'
    
        files = []
        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            files.append(key)
        return files
        

    
    def get_folder(self, s3_folder: str):
        """Download contents from a folder from S3"""
        folder = f'{self.username}/{self.s3_sub_folder}/{s3_folder}/'

        if not os.path.exists(s3_folder):
            os.makedirs(s3_folder)

        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            file_name = s3_folder + '/' + key[key.find(s3_folder + '/') + len(s3_folder) + 1:]
            self.client.download_file(self.bucket, key, file_name)

    
    def clear_local_folder(self, local_folder: str, count: int=0):
        """Function to clear all of the contents of a folder in the local project"""
        try:
            for root, dirs, files in os.walk(local_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        except: 
            # check not stuck in loop
            if count <= 3:
                self.clear_local_folder(local_folder, count + 1)

    