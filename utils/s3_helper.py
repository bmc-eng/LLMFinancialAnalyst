import boto3
import os

class S3Helper:
    """
    Helper class to handle files in a folder moving to S3 storage
    in the Bloomberg Lab sandbox. Initialize with a project folder name for S3
    project_folder: str - sub folder in Bloomberg Lab to save project data
    """
    
    def __init__(self, project_folder: str):
        self.username = os.environ['BQUANT_USERNAME']
        self.project_folder = project_folder
        self.bucket = os.environ['BQUANT_SANDBOX_USER_BUCKET']
        self.client = boto3.client("s3")

    def _list_local_files(self, local_folder:str) -> list[str]:
        """
        Internal function to list all of the files in the local folder
        """
        file_directory = []
        for root, dirs, files in os.walk(local_folder, topdown=False):
            for name in files:
                
                file_directory.append(os.path.join(root, name))
            for name in dirs:
                pass
        return file_directory
    
    
    def add_folder(self, local_folder: str, s3_folder: str):
        """
        Upload all files in a local folder in a folder to S3
        local_folder: str - the folder in the local project directory
        s3_folder: str - the sub-folder to save in S3 storage
        """
        # Walk through each of the files in the folder structure
        files = self._list_local_files(local_folder)
        for file in files:
            # remove the local folder name
            obj_name = f'{self.username}/{self.project_folder}/{s3_folder}{file[len(local_folder):]}'
            res = self.client.upload_file(file, self.bucket, obj_name)
        print(f'Uploaded: {local_folder}')

    
    def add_file(self, local_filename: str, s3_folder:str = None):
        """
        Upload a file from the project directory to S3
        local_filename: str - the file to upload from local storage
        s3_folder: str - the folder to save the file in Bloomberg Lab S3
        """
        # check to see if the file is part of a file structure
        if local_filename.find('/') >=0:
            file_dir = local_filename.split('/')
            filename = file_dir[-1]
        else:
            filename = local_filename
        # Create the S3 folder path
        if s3_folder == None:
            obj_name = f'{self.username}/{self.project_folder}/{filename}'
        else:
            obj_name = f'{self.username}/{self.project_folder}/{s3_folder}/{filename}'
        res = self.client.upload_file(local_filename, self.bucket, obj_name)
        return res

    
    def delete_file(self, filename: str):
        """
        Delete a single file
        filename: str - file name in Bloomberg Lab S3 to delete. Includes the subfolder path
        """
        obj_name = f'{self.username}/{self.project_folder}/{filename}'
        self.client.delete_object(Bucket=self.bucket, Key=obj_name)
        print(f"Deleted: {filename}")
    
    
    def delete_folder(self, s3_folder:str):
        """
        Delete all files in a folder
        s3_folder: str - the folder in Bloomberg Lab S3 to delete
        """
        files = self.list_folder(s3_folder)
        for file in files:
            self.client.delete_object(Bucket=self.bucket, Key=file)

    
    def list_folder(self, s3_folder:str = None) -> list[str]:
        """
        List all of the files in an s3 folder.
        s3_folder: str - the sub folder in Bloomberg Lab S3 storage
        Return: List of files
        """
        if s3_folder != None:
            folder = f'{self.username}/{self.project_folder}/{s3_folder}'
        else:
            folder = f'{self.username}/{self.project_folder}'
            
        files = []
        try:
            for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
                key = file['Key']
                files.append(key)
        except KeyError:
            # the folder is empty
            pass
        return files
        
    
    def get_file(self, filename:str, local_filename: str = None):
        """
        Download a file from Bloomberg Lab S3
        filename: str - file in S3 to download. Remove the project folder name before passing
        """
        obj_name = f'{self.username}/{self.project_folder}/{filename}'
        if local_filename == None:
            local_filename = filename
        self.client.download_file(self.bucket, obj_name, local_filename)

    
    def get_folder(self, s3_folder: str):
        """Download contents from a folder from S3"""
        folder = f'{self.username}/{self.project_folder}/{s3_folder}/'

        if not os.path.exists(s3_folder):
            os.makedirs(s3_folder)

        for file in self.client.list_objects(Bucket=self.bucket, Prefix=folder)['Contents']:
            key = file['Key']
            file_name = s3_folder + '/' + key[key.find(s3_folder + '/') + len(s3_folder) + 1:]
            self.client.download_file(self.bucket, key, file_name)

    
    def clear_local_folder(self, local_folder: str, count: int=0):
        """Function to clear all of the contents of a folder in the local project
        local_folder: str - the local file directory in the project folder to clear
        count: int (Optional) - count to clear for multi-threaded applications"""
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

    