# LLM Financial Analyst

Dissertation project for Computer Science MSc. This project will investigate the use of LLMs for financial analysis of equities on the SPX and FTSE 100 companies.

## Build Steps

### Get Data
V1 is complete, gets the data and pivots into a format that can be converted into a prompt easily. To do:
- Get the reported date to make sure this is point-in-time


### Company Object
NEED TO BUILD: Each company should have its own object - this will have the IS, BS and pricing data for the security + the revision date for the datasets to be used when adding. This will be used to build the prompts.

### Rebuilding .gitignore file
The following files should be added into the .gitignore file:

```
# Folders
.Trash-10000
.ipynb_checkpoints
.virtual_documents
.__pycache__
__pycache__
Data

# Files
pass.txt
.env
.project.json
project.json
```