# LLM Financial Analyst

Dissertation project for Computer Science MSc. This project will investigate the use of LLMs for financial analysis of equities on the Dow Jones and FTSE 100 companies. 

1. Inference via Open Source and Frontier Models (via AWS Bedrock) for financial statement analysis
2. Construction of an events backtester built on top of Bloomberg Signal Lab libraries
3. Comparison of models to the base
4. Creation of an agentic model that takes into account news and industry reports (also fine tune an open source model for financial statement analysis)

Agentic model will have: Earnings direction, drivers of earnings analysis from industry reports, peer group analysis, stock valuation model. This will make a prediction on the whether the stock is valued correctly.

## Research Objectives
To build on the work of Kim et al (2024), Xu et al (2025), Zhang, Zhao et al (2024) to produce a multi-agentic financial analyst system that is backtested and creates long-term (3 month) investment decisions. 

## Running the Project
This project requires Bloomberg Lab for Enterprise to run. It will require Textual Analytics for the News datasets, Signal Lab for the Portfolio Analytics and access to Bedrock. 

## Deliverables
The deliverables for this project are:

- A series of Jupyter Notebooks highlighting the research and steps used to construct the models:
    - 01A - PIT datasets -> This is used to get all of the point-in-time financial data
    - 01B - DataPacks -> This is used to get datasets needed to run the strategy backtester
    - 03A - Multi-GPU -> This generates the strategies from the open source LLMs

- Data Module (requesters)
    - Data Requester to retrieve and process point-in-time financial statement and company reference datasets
    - News Requester to retrieve all company news over the backtest time horizon
- Model Module (models)
    - model_helper.py to help request, store and load Huggingface models
    - model finetuner to help fine tune open source models
- Agent Module (agents)
    - FinancialAnalystAgent to run financial analysis tasks on a company with financial statement datasets and news datasets
    - CommitteeAgent to debate the analyst reports and put forward alternative investment thesis
- Strategy Construction Module 
    - model_inference.py to run multi-GPU inference on Huggingface models and generate a strategy
    - model finetune inference to help run inference tasks on the finetuned models
    - prompts to record all of the system prompts used
- Strategy Analysis Module
    - event_study.py is an Event Backtester to test the financial outcome of a strategy


- Python utilities of helper functions to:
    - Logger for model outputs 
    - Storage helper to store information in Bloomberg Lab S3
    - Construction of portfolio legs


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


## Model Development Notes

#### 24th Apr - Successful run of the Agentic backtest

#### 23rd Apr - Added the Agentic debate format
Added the debate format into the test example. Changed and modified the prompts slightly to build consensus and avoid the conversation going round and round in circles. Implemented as .py file in object format

#### 22nd Apr - Added first version of Agentic model
Added the flow of the agentic process. Rerun of the fine-tuned model

#### 21st Apr - News request class + Agentic model
Created the ability to request Bloomberg News headlines. Updated the Model Helper class. 

#### 20th Apr - Run of fine-tuned models
Created inference module for fine tuned model - needed to create a separate class for this to continue to support PEFT. Experienced errors in the multi-GPU inference runs when including the PEFT library. It is possible that this is not supported from a multi-threading perspective when using notebook_launcher. Investigation of this is out of scope for the current project.

Rationalised the S3 Helper and ModelHelper objects to inherit from each other. 

#### 3rd Apr - Model changed to guess the direction of earnings


#### 23rd Mar - Started work on Bedrock
Run a multi-agentic approch with Bedrock to get deep thought and structured output. Have run with Claude but the results have not been very good compared to the Llama 3M model. Have changed the temperature and filtered to remove less confident recommendations to see if this will improve. Running another model with a slightly lower temperature 0.7 >> 0.5. Each model is taking around 10 mins to run.

#### 19th Mar - OpenAI results run in Langgraph
Took 2hours 23 mins to run on 896 prompts in OpenAI. The results as well were not very good. Qwen model results were also very poor compared to Llama for open source.

#### 14th Mar - Run model with Qwen but getting issues with CoT
Running into a strange error in the multi-threading code that have not been able to work out how to resolve. It is throwing the error after 1:30h so very difficult to replicate. W0314 09:47:59.334846 336 site-packages/torch/multiprocessing/spawn.py:169] Terminating process 371 via signal SIGTERM Updating logic to try to resolve this.

#### 10th Mar - Completed strategy analysis
Added each of the strategies into the events backtester and plotted their results. The results are suggesting that the open source models do not beat the base line of the consensus sellside analyst for the Dow Jones. 

#### 9th Mar - Built the EventBacktester
Have completed the EventBacktest class to help with evaluating each of the strategies. This backtester takes a list of trades and constructs a portfolio based on the trade (long/ short). It does this by modifying the portfolio weights and ignoring the signals that are usually used in the Signal Lab backtester. The EventBacktester is a wrapper that sits on top of Bloomberg Signal Lab. 

Tested the base test case of consensus recommendations and also the first run of the Llama model. This can now be used to test enhancements to the model.

#### 24th Feb - Additional runs of the model with Multi-GPU
Including running both zero shot and chain of thought prompting with Llama, Qwen and Deepseek.

#### 21st Feb - Refactored the Model inference code for Multi-GPU
Refactoring of codebase to allow for multi-GPU inference and wrapped in a class. Created a inference object to record the dataset, open source model used and the system prompt so that comparisons can be made in a few weeks time once the event backtester is completed. Have run 4 sets of results which are now much faster thanks to using 4 GPU cores. 

#### 14th Feb - Three runs of the model to date
Issue with the larger models. Running into out of memory issues and also losing track of the model run results. This is due to the model only running on one GPU rather than the up to 4 GPUs that have access to in Bloomberg Lab. Will need to do some investigation to see how easy this will be to run multi-GPU inference.

#### 4th Feb - Explored vllm and tensor_parallel
Changed the logic for generating the prompt to reduce the token size of the prompt. It is now 4,257 vs 5,165 before reduction. 
Checked out:
- https://medium.com/tr-labs-ml-engineering-blog/tensor-parallel-llm-inferencing-09138daf0ba7
- https://www.kaggle.com/code/blacksamorez/tensor-parallel-int4-llm/

#### 3rd Feb - Single security run of the model
Using Llama 3.3 3B parameter model with no fine tuning. Number of tokens in the prompt 5,165 and need to reduce this. Considering changing the number of fields that are included in the prompt to speed this up. Also need to look at multi-gpu and splitting inference across multiple gpus to be able to test larger models.


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