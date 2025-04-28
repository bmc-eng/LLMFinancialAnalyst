###########################
####   Introduction #######
###########################

# This module details the prompts and code needed to run the Investment Committee Agent. 
# 



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

from typing import Dict, TypedDict, Optional, Annotated


###########################
######   PROMPTS ##########
###########################

debate_agent_initial_system_prompt = """You are part of the investment committee at an asset management firm. Your senior analyst in the {sector} sector has presented their opinion on the future direction of earnings for blah. With your fellow committee members, discuss whether you now think the stock is undervalued and should be a BUY or overvalued and should be a SELL. Use DCF analysis and other methods to value the stock. Show your workings. You are advocating to {decision} the stock for the next quarter. Give a short 200 word summary of your decision. Only use the following context. Do not use any other information. Think through the response. Context: {senior_analyst_report} {financial_statement} {stock_prices}"""


debate_agent_system_prompt = """You are part of the investment committee at an asset management firm. Your senior analyst in the {sector} sector has presented their opinion of blah to you and the fellow committee members. You must now debate on whether the stock price is undervalued or overvalued. You previously voted to {decision}. Based on the prior conversation, comment if you agree with your previous decision or want to change. You should only vote to SELL if your calculations show the stock is significantly overvalued and you believe a correction in the price will happen over the next quarter. You should vote to BUY if the stock is undervalued or flat. If you want to change you must explain your reason and show your workings. Do not use any information other than in the analyst report or the previous conversation. Provide the argument in less than 150 words. Think through your analysis. For review: {senior_analyst_report} \nPrior conversation: {conversation}"""


debate_direction_system_prompt = """Based on the response, return the sentiment of the response as BUY, SELL or HOLD. Only return a single word: BUY, SELL or HOLD. Conversation: {conversation}"""


result_system_prompt = """You are part of the investment committee at an asset management firm. You are responsible for recording the decision of the investment committee. You must record the majority decision of the analysts to BUY, SELL or HOLD the security. If there is no majority decision, you hold the deciding vote. If there is a majority decision in the voting pattern, you do not change this decision. The voting pattern is: {voting_pattern} The senior analyst report is: {senior_analyst_report} \nThe debate is: {conversation} \n"""

###########################
######   LLM IDS ##########
###########################

model_claude_id = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'
model_claude_small = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
model_llama_id = 'us.meta.llama3-1-70b-instruct-v1:0'
model_llama_small_id = "us.meta.llama3-1-8b-instruct-v1:0"

class Recommendation(BaseModel):
    recommendation: str = Field(..., description="BUY, SELL or HOLD. One value only")
    reason: str = Field(..., description="Summary of the final decision from the committee")
    confidence: str = Field(..., description="How confident you are of the decision")


class CommitteeState(TypedDict):
    classification: Optional[str] = None
    last_agent: Optional[str] = None
    history: Annotated[list, add_messages]
    summary_buy: Optional[str] = None
    summary_sell: Optional[str] = None
    summary_hold: Optional[str] = None
    current_response: Optional[str] = None
    count: Optional[int] = None
    results: Optional[str] = None
    consensus: Optional[dict[str,str]] = None
    senior_analyst_report: Optional[str] = None
    financial_statement_analysis: Optional[str] = None
    sector: Optional[str] = None
    stock_prices: Optional[str] = None
    


class CommitteeAgent():
    """
    Class to define the committee debating agents. These agents will define the 
    """

    def __init__(self):
        """
        Constructor for CommitteeAgent.
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
        self._llm_debate = ChatBedrock(
            client = boto3_bedrock,
            model_id = model_llama_id,
            temperature = 0.01,
            max_tokens = 4000,
            rate_limiter = rate_limiter
        )

        self._llm_report_writer = ChatBedrock(
            client = boto3_bedrock,
            model_id = model_claude_small,
            temperature = 0.01,
            max_tokens = 4000,
            rate_limiter = rate_limiter
        ).with_structured_output(Recommendation)

        # set up the Prompt Templates
        self.agentic_initial_template = PromptTemplate.from_template(debate_agent_initial_system_prompt)
        self.agentic_debate_template = PromptTemplate.from_template(debate_agent_system_prompt)
        self.debate_direction_template = PromptTemplate.from_template(debate_direction_system_prompt)
        self.results_template = PromptTemplate.from_template(result_system_prompt)

        self.committee_workflow = StateGraph(CommitteeState)

        # add the nodes
        self.committee_workflow.add_node('next_node', self._next_node)
        self.committee_workflow.add_node('handle_buy', self._handle_buy)
        self.committee_workflow.add_node('handle_sell', self._handle_sell)
        self.committee_workflow.add_node('handle_hold', self._handle_hold)
        self.committee_workflow.add_node('result', self._result)

        # add the edges to the nodes
        self.committee_workflow.add_conditional_edges(
            "next_node",
            self._decide_next_node,
            {
                "handle_buy": "handle_buy",
                "handle_sell": "handle_sell",
                "handle_hold": "handle_hold",
                "result": "result"
            }
        )

        self.committee_workflow.add_conditional_edges(
            "handle_buy",
            self._check_conv_length,
            {
                "result": "result",
                "next_node": "next_node"
            }
        )
        
        self.committee_workflow.add_conditional_edges(
            "handle_sell",
            self._check_conv_length,
            {
                "result": "result",
                "next_node": "next_node"
            }
        )
        
        self.committee_workflow.add_conditional_edges(
            "handle_hold",
            self._check_conv_length,
            {
                "result": "result",
                "next_node": "next_node"
            }
        )

        # set entry point and end point
        self.committee_workflow.set_entry_point("handle_buy")
        self.committee_workflow.add_edge('result', END)

        # Compile the LangGraph app
        self.app = self.committee_workflow.compile()
        
    
    def run(self, senior_analyst_report:str, 
            financial_statement_analysis: str, security_data: dict):
        """
        Entry point into the Agent. 
        senior_analyst_report: str - a summary report from the FinancialAnalysisAgent
        financial_statement_analysis: str - the initial report on the 
        """
        initial_state = {
            'count':0,
            'history':['Nothing'],
            'current_response':'',
            'summary_buy': 'Nothing',
            'summary_sell': 'Nothing',
            'summary_hold': 'Nothing',
            'consensus': {'BUY':'BUY', 'SELL':'SELL', 'HOLD':'HOLD'},
            'senior_analyst_report': senior_analyst_report,
            'financial_statement_analysis': financial_statement_analysis,
            'sector': security_data['sector'],
            'stock_prices': security_data['stock_price']
        }

        return self.app.invoke(initial_state)
        
    def get_graph(self):
        """
        Function to return a Mermaid visual of the Agent
        """
        return self.app.get_graph()
    
    def _llm_debate_invoke (self, prompt: str) -> str:
        """
        Function to invoke the debate LLM
        Return: str - response from LLM
        """
        return self._llm_debate.invoke(prompt).content

    
    def _debate_format(self, state, state_update, decision):
        """
        Function to run the debate between three agents. All use the same template/ format
        state_update: str - pass name of object to store the agent summary
        decision: str - identity of the agent
        """
        summary = state.get('history')#.strip()
        summary_x = state.get(state_update, '').strip()
        current_response = state.get('current_response', '').strip()

        # Get company specific datasets
        senior_analyst_report = state.get('senior_analyst_report')
        sector = state.get('sector')
        
        if summary_x=='Nothing':
            # this is the initial argument
            # get additional information on the security
            financial_statement_analysis = state.get('financial_statement_analysis')
            stock_prices = state.get('stock_prices')

            # Create the summary prompt for initial arguments
            prompt_in = self.agentic_initial_template.format(decision=decision, 
                                                        senior_analyst_report=senior_analyst_report,
                                                        financial_statement=financial_statement_analysis,
                                                        stock_prices=stock_prices,
                                                        sector=sector)
            # Call the LLM and record the argument
            argument = decision + ":" + self._llm_debate_invoke(prompt_in)

            return {'history': [argument], 
                        state_update: argument, 
                        'current_response': argument, 
                        'count':state.get('count')+1,
                        'last_agent': decision}

        else:
            # this runs during the debate after the analysts have presented their opening arguments
            # From this point the agents can change their decision
            current_consensus = state.get('consensus')
            prompt_in = self.agentic_debate_template.format(decision=current_consensus[decision],
                                                          sector=sector,
                                                          senior_analyst_report=senior_analyst_report,
                                                          conversation=summary)
            argument = decision + ":" + self._llm_debate_invoke(prompt_in)
    
            return {'history': [argument],#'history': summary + '\n' + argument,
                    'current_response': argument, 
                    'count': state.get('count') + 1,
                    'last_agent': decision}
    
    def _next_node(self, state):
        last_agent = state.get('last_agent').strip()
        response_from_agent = state.get('current_response')
        current_consensus = state.get('consensus')
        prompt_in = self.debate_direction_template.format(conversation=response_from_agent)
        last_node = self._llm_debate_invoke(prompt_in)
        #print(last_agent + ":" + last_node)
        current_consensus[last_agent] = last_node
        return {'consensus': current_consensus}
    
    def _handle_buy(self, state):
        return self._debate_format(state, 'summary_buy', 'BUY')
    
    def _handle_sell(self, state):
        return self._debate_format(state, 'summary_sell', 'SELL')
    
    def _handle_hold(self, state):
        return self._debate_format(state, 'summary_hold', 'HOLD')
    
    def _result(self, state):
        summary = state.get('history')#.strip()
        senior_report = state.get('senior_analyst_report')
        voting = state.get('consensus')
        prompt_in = self.results_template.format(conversation=summary, voting_pattern=list(voting.values()), senior_analyst_report=senior_report)
        #structured_llm = self._llm_debate.with_structured_output(Recommendation)
        return {"results": self._llm_report_writer.invoke(prompt_in)}

    
    def _decide_next_node(self, state):
        last_agent = state.get('last_agent')
        current_consensus = state.get('consensus')
        count = state.get('count')
        if state.get('summary_sell', '') == 'Nothing':
            return 'handle_sell'
        if state.get('summary_hold', '') == 'Nothing':
            return 'handle_hold'

        # If there is a majority decision then end the process
        if count % 3 == 0 and len(set(current_consensus.values())) <= 2:
            # all the agents are in consensus
            return 'result'
        else:
            # agent is agreeing with their classification.
            # debate should continue
            if last_agent == 'BUY':
                return 'handle_sell'
            if last_agent == 'SELL':
                return 'handle_hold'
            if last_agent == 'HOLD':
                return 'handle_buy'

    
    def _check_conv_length(self, state):
        return "result" if state.get("count")==9 else "next_node"

        

