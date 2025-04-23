import boto3
import botocore
import random
import json

from langchain.prompts import PromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from typing import Dict, TypedDict, Optional


statement_analysis_system_prompt = """You are a financial analyst. Use the following income statement and balance sheet to make a decision on if earnings will increase over the next financial period. Think step-by-step through the financial statement analysis workflow. Your report should have the following sections: 1. Analysis of current profitability, liquidity, solvency and efficiency ratios; 2. time-series analysis across the ratios; 3. Analysis of financial performance; 4. Stock Price analysis; 5. Decision Analysis looking at the positive and negative factors as well as the weighting in the final decision; 6. Final Decision. Make your decision only on the datasets. 7. Provide a breakdown of information that you wished was included to make a better decision. Explain your reasons in less than 250 words. Indicate the magnitude of the increase or decrease. Provide a confidence score for how confident you are of the decision. If you are not confident then lower the confidence score. {financials}"""

clean_headlines_system_prompt = """You are an assistant to a financial analyst analyzing {security} You must remove any reference to {security} and their products from the following list of headlines and replace them with the term 'blah'. Replace the names of any people such as ceo in the article with the term 'whah' do not refer to {security} at all in your answer:{headlines}"""


company_news_system_prompt = """You are a financial analyst and are reviewing news for company called blah over the last three months. Blah is in the {sector} sector. Start by listing the revenue drivers for the sector. Then look through the below headlines and determine if blah will see an increase or decrease in their earnings over the next quarter. Think through your response. {headlines}"""


senior_analysis_prompt = """You are a senior financial analyst and review your teams work. You are looking at a financial summary and news for 'blah'. Using the summaries only, critique the report and construct an alternative narrative. If the narrative is in agreement with the two reports, make clear your belief in the direction of earning. If in disagreement, state why you disagree. Think through your response. {financial_summary} \n {news_summary}"""



class CompanyData(TypedDict):
    name: str
    figi_name: str
    sector: str
    sec_fs: str
    headlines: list[str]
    stock_prices: str


class AnalystState(TypedDict):
    company_details: Optional[CompanyData]
    initial_analysis: Optional[str]
    cleaned_headlines: Optional[list]
    news_report: Optional[str]
    senior_report: Optional[str]


class FinancialAnalystAgent:

    """
    FinancialAnalystAgent class: This takes a single security and puts the security through a review to 
    determine if this is a good fit for investment over the next quarterly period. 
    """
    def __init__(self):
        """
        Constructor for FinancialAnalystAgent. Requires information on a single company
        """
        # configure the LLMs
        
        
        
        # Create the prompt templates
        self.statement_analysis_template = PromptTemplate.from_template(statement_analysis_system_prompt)
        self.clean_headlines_template = PromptTemplate.from_template(clean_headlines_system_prompt)
        self.company_news_template = PromptTemplate.from_template(company_news_system_prompt)
        self.senior_analyst_template = PromptTemplate.from_template(senior_analysis_prompt)
        

        # set up the LangGraph workflow
        self.analyst_workflow = StateGraph(AnalystState)

        # add each of the nodes
        self.analyst_workflow.add_node('financial_statement_analysis', self.financial_statement_analysis)
        self.analyst_workflow.add_node('clean_headlines', self.clean_headlines)
        self.analyst_workflow.add_node('news_summary', self.news_summary)
        self.analyst_workflow.add_node('final_report', self.final_report)

        # add the node edges
        self.analyst_workflow.set_entry_point('financial_statement_analysis')
        self.analyst_workflow.add_edge('financial_statement_analysis', 'clean_headlines')
        self.analyst_workflow.add_edge('clean_headlines', 'news_summary')
        self.analyst_workflow.add_edge('news_summary', 'final_report')
        self.analyst_workflow.add_edge('final_report', END)

        self.app = self.analyst_workflow.compile()


    def _analyst_llm(self, llm, prompt):
        """
        Function to call an LLM with the prompt
        """
        return llm.invoke(prompt).content

    
    def run(self, security_data, news_data, as_of_date):
        pass

    
    def financial_statement_analysis(self, state):
        # Create the prompt to feed into the model
        company_details = state.get('company_details')
        sec_fs = company_details['sec_fs']
        prompt_in = self.statement_analysis_template.format(financials=sec_fs)
        financial_analysis = self._analyst_llm(llm_thinker, prompt_in)
        return {'initial_analysis': financial_analysis}

    
    def clean_headlines(self, state):
        company_details = state.get('company_details')
        unclean_headlines = company_details['headlines']
        # Create the prompt to feed into the model
        prompt_in = self.clean_headlines_template.format(headlines=unclean_headlines, security=name)
        clean_headlines = self._analyst_llm(llm_small, prompt_in)
        return {'cleaned_headlines': clean_headlines}

    
    def news_summary(self, state):
        # Create the prompt to feed into the model
        company_details = state.get('company_details')
        clean_headlines = state.get('cleaned_headlines')
        prompt_in = self.company_news_template.format(headlines=clean_headlines[1:], sector=company_details['sector'])
        news_summarisation = self._analyst_llm(llm_thinker, prompt_in)
        return {'news_summary': news_summarisation}

    
    def final_report(self, state):
        company_details = state.get('company_details')
        initial_analysis = state.get('initial_analysis')
        news_summary = state.get('news_summary')
        prompt_in = self.senior_analyst_template.format(financial_summary=initial_analysis, news_summary=news_summary)
        final_report_output = self._analyst_llm(llm_thinker, prompt_in)
        return {'senior_report': final_report_output} 