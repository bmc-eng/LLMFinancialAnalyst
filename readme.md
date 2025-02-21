# LLM Financial Analyst

Dissertation project for Computer Science MSc. This project will investigate the use of LLMs for financial analysis of equities on the SPX, Dow Jones and FTSE 100 companies.

## Build Steps

### Get Data
V1 is complete, gets the data and pivots into a format that can be converted into a prompt easily. To do need a better V2 version that is truely point in time:

- Get a list of reporting dates by company
- Create a company specific object
    - name
    - reporting date []
        - income statement (-5 periods)
        - balance sheet (-5 periods)
        - pricing (-1Y daily)
    - prompt
        - result []
    - populate_data()

- Create a list of trades by date, company and buy/sell/ hold based on LLM output
- Get the reported date to make sure this is point-in-time


### Model Development Notes

#### Hitting out of memory issues with inference tasks
Refactoring of codebase to allow for multi-GPU inference


#### Three runs of the model to date - 14th Feb


#### Explored vllm and tensor_parallel - 4th Feb
Changed the logic for generating the prompt to reduce the token size of the prompt. It is now 4,257 vs 5,165 before reduction. 
Checked out:
- https://medium.com/tr-labs-ml-engineering-blog/tensor-parallel-llm-inferencing-09138daf0ba7
- https://www.kaggle.com/code/blacksamorez/tensor-parallel-int4-llm/

#### Single security run of the model - 3rd Feb 2025
Using Llama 3.3 3B parameter model with no fine tuning. Number of tokens in the prompt 5,165 and need to reduce this. Considering changing the number of fields that are included in the prompt to speed this up. Also need to look at multi-gpu and splitting inference across multiple gpus to be able to test larger models.



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