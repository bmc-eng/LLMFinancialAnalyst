import boto3
import botocore
import random
import json
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_aws import ChatBedrock
from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from typing import Dict, TypedDict, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta

###########################
######   PROMPTS ##########
###########################


statement_analysis_system_prompt = """You are a financial analyst. Use the following income statement and balance sheet to decide if earnings will increase over the next financial period. Think step-by-step through the financial statement analysis workflow. Your report should have the following sections: 1. Analysis of current profitability, liquidity, solvency and efficiency ratios. State the formula and then show your workings.\n 2. time-series analysis across the ratios\n 3. Analysis of financial performance\n 4. weigh up the positive and negative factors\n 5. Final Decision. Make your decision only on the analysis you have done. Explain your reasons in less than 250 words. Indicate the magnitude of the increase or decrease. Provide a confidence score for how confident you are of the decision. If you are not confident then lower the confidence score. {financials}"""

clean_headlines_system_prompt = """You are an assistant to a financial analyst analyzing {security} You must remove any reference to {security} and their products from the following list of headlines and replace them with the term 'blah'. Replace the names of any people such as ceo in the article with the term 'whah' do not refer to {security} at all in your answer:{headlines}"""


company_news_system_prompt = """You are a financial analyst and are reviewing news for company called blah over the last three months. Blah is in the {sector} sector. Start by listing the revenue drivers for the sector. Then look through the below headlines and determine if blah will see an increase or decrease in their earnings over the next quarter. Think through your response. {headlines}"""


senior_analysis_prompt = """You are a senior financial analyst and review your teams work. You are looking at a financial summary, financial statements and news report for 'blah'. Using the summaries and the financial statements only, review both reports and decide if you agree with the narrative of earnings increase or decrease. If your narrative is in agreement with the two reports, make clear your belief in the direction of earnings over the next quarter. If in disagreement, state why you disagree. Think through your response. Analyst Summary: {financial_summary} \n News Summary: {news_summary} \n Financial Statements: {financial_statements}"""

analyst_writer_prompt = """You are an assistant to a financial analyst. You are responsible for formatting the documents that the analyst produces into a machine readable format. Use only the information provided in the context. Convert it into the structured output. Do not add anything to the analysts report and do not change the recommendation. Do not hallucinate. Find the investment decision. Find the conclusion. Add all of the wording of the thought process into the steps section. context: {context}"""


###########################
######   LLM IDS ##########
###########################

model_claude_id = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
model_claude_small = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
model_llama_id = 'us.meta.llama3-1-70b-instruct-v1:0'
model_llama_small_id = "us.meta.llama3-1-8b-instruct-v1:0"

class CompanyData(TypedDict):
    name: str
    figi_name: str
    sector: str
    sec_fs: str
    headlines: list[str]
    stock_prices: str


class EarningsReportOutput(BaseModel):
    """Class used by Structured Output to format into JSON"""
    direction: str = Field(..., description="earnings will increase or decrease or stay flat")
    magnitude: str = Field(..., description="size of the increase or decrease")
    reason: str = Field(..., description="Summary of the final decision")
    confidence: str = Field(..., description="How confident you are of the decision")

class AnalystState(TypedDict):
    """Class used by LangGraph to record State"""
    company_details: Optional[CompanyData]
    financial_report: Optional[str]
    cleaned_headlines: Optional[list]
    news_report: Optional[str]
    senior_report: Optional[str]
    final_output: Optional[EarningsReportOutput]


class FinancialAnalystAgent:

    """
    FinancialAnalystAgent class: This takes a single security and puts the security through a review to 
    determine if this is a good fit for investment over the next quarterly period. 
    """
    def __init__(self):
        """
        Constructor for FinancialAnalystAgent. Requires information on a single company
        """
        # configure connectivity to Bedrock
        config = botocore.config.Config(read_timeout=1000)
        boto3_bedrock = boto3.client('bedrock-runtime', config=config)

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=50,
            check_every_n_seconds=1,
            max_bucket_size=10,
        )

        # Configure the LLMs
        self.llm_thinker = ChatBedrock(
            client = boto3_bedrock,
            model_id = model_claude_id,
            temperature = 0.01,
            max_tokens=4000,
            rate_limiter = rate_limiter
        )

        self.llm_small = ChatBedrock(
            client = boto3_bedrock,
            model_id = model_llama_small_id,
            temperature = 0.01,
            max_tokens = 4000,
            rate_limiter = rate_limiter
        )

        self.llm_report_writer = ChatBedrock(
            client = boto3_bedrock,
            model_id = model_claude_small,
            temperature = 0.01,
            max_tokens = 4000,
            rate_limiter = rate_limiter
        ).with_structured_output(EarningsReportOutput)
        
        
        # Create the prompt templates
        self.statement_analysis_template = PromptTemplate.from_template(statement_analysis_system_prompt)
        self.clean_headlines_template = PromptTemplate.from_template(clean_headlines_system_prompt)
        self.company_news_template = PromptTemplate.from_template(company_news_system_prompt)
        self.senior_analyst_template = PromptTemplate.from_template(senior_analysis_prompt)
        self.analyst_assistant_template = PromptTemplate.from_template(analyst_writer_prompt)
        

        # set up the LangGraph workflow
        self.analyst_workflow = StateGraph(AnalystState)

        # add each of the nodes
        self.analyst_workflow.add_node('financial_statement_analysis', self._financial_statement_analysis)
        self.analyst_workflow.add_node('clean_headlines', self._clean_headlines)
        self.analyst_workflow.add_node('news_summary', self._news_summary)
        self.analyst_workflow.add_node('final_report', self._final_report)
        self.analyst_workflow.add_node('structured_report', self._structured_report)

        # add the node edges
        self.analyst_workflow.set_entry_point('financial_statement_analysis')
        self.analyst_workflow.add_edge('financial_statement_analysis', 'clean_headlines')
        self.analyst_workflow.add_edge('clean_headlines', 'news_summary')
        self.analyst_workflow.add_edge('news_summary', 'final_report')
        self.analyst_workflow.add_edge('final_report', 'structured_report')
        self.analyst_workflow.add_edge('structured_report', END)

        self.app = self.analyst_workflow.compile()


    def _analyst_llm(self, llm, prompt):
        """
        Function to call an LLM with the prompt
        """
        return llm.invoke(prompt).content
    
    
    def run(self, security_data: dict, news_data: pd.DataFrame, as_of_date: str) -> AnalystState:
        """
        Run the agent with a security on a particular date
        security_data: CompanyData - information from the company on the as_of_date
        news_data: pd.DataFrame - all of the Bloomberg News data for the security universe
        as_of_date: str - as of date for the analysis to prevent lookahead bias
        """
        # Filter the news first to get the company news filtered by security and date
        filtered_news = self._filter_news_by_company_by_date(news_data, security_data['figi'], as_of_date)
        # Set up the state to begin the analysis
        company_details = CompanyData({'name':security_data['name'],
                              'figi_name': security_data['figi'],
                              'sector': security_data['sector'],
                              'sec_fs': security_data['sec_fs'],
                              'headlines': filtered_news,
                              'stock_prices': security_data['stock_price']})
        return self.app.invoke({'company_details': company_details})


    def get_graph(self):
        return self.app.get_graph()

    
    def _financial_statement_analysis(self, state):
        # Create the prompt to feed into the model
        company_details = state.get('company_details')
        sec_fs = company_details['sec_fs']
        prompt_in = self.statement_analysis_template.format(financials=sec_fs)
        financial_analysis = self._analyst_llm(self.llm_thinker, prompt_in)
        return {'financial_report': financial_analysis}

    
    def _clean_headlines(self, state):
        company_details = state.get('company_details')
        unclean_headlines = company_details['headlines']
        name = company_details['name']
        # Create the prompt to feed into the model
        prompt_in = self.clean_headlines_template.format(headlines=unclean_headlines, security=name)
        clean_headlines = self._analyst_llm(self.llm_small, prompt_in)
        return {'cleaned_headlines': clean_headlines}

    
    def _news_summary(self, state):
        # Create the prompt to feed into the model
        company_details = state.get('company_details')
        clean_headlines = state.get('cleaned_headlines')
        prompt_in = self.company_news_template.format(headlines=clean_headlines[1:], sector=company_details['sector'])
        news_summarisation = self._analyst_llm(self.llm_thinker, prompt_in)
        return {'news_report': news_summarisation}

    
    def _final_report(self, state):
        company_details = state.get('company_details')
        financial_report = state.get('financial_report')
        news_report = state.get('news_report')
        sec_fs = company_details['sec_fs']
        prompt_in = self.senior_analyst_template.format(financial_summary=financial_report, 
                                                        news_summary=news_report,
                                                       financial_statements=sec_fs)
        final_report_output = self._analyst_llm(self.llm_thinker, prompt_in)
        return {'senior_report': final_report_output} 

    def _structured_report(self, state):
        """Final step to structure the report into something ready for output"""
        senior_analyst_report = state.get('senior_report')
        prompt_in = self.analyst_assistant_template.format(context=senior_analyst_report)
        structured_output = self.llm_report_writer.invoke(prompt_in)

        return {'final_output': structured_output}

    def _filter_news_by_company_by_date(self, news_dataset, security, max_date = None):
        
        if max_date != None:
            #calaculate the minimum date to get a 3 month window
            min_date = datetime.strptime(max_date,"%Y-%m-%d") + relativedelta(months=-3)
            min_date = min_date.strftime("%Y-%m-%d")
            filtered_dataset = news_dataset[(news_dataset['TimeOfArrival'] < max_date) &  (news_dataset['TimeOfArrival'] >= min_date) & (news_dataset['Assigned_ID_BB_GLOBAL'] == security)]
        else:
            filtered_dataset = news_dataset[news_dataset['Assigned_ID_BB_GLOBAL'] == security]
        
        filtered_list = filtered_dataset['Headline'].to_list()
    
        if len(filtered_list) >= 50:
            return random.sample(filtered_list, 50)
        else:
            return filtered_list