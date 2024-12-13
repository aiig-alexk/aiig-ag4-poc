import numpy as np
import pandas as pd

"""
economic_analyst.py

This module contains functions and classes for economic analysis.

simple test to mimic risk analytic job. 
the analyst agent uses other junior agents (data, analytic, news, etc.) to complete the report
first iteration , copied from autogen LiteratureReview sample: https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/examples/literature-review.html

"""

# Import necessary libraries
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

# wrap the function into the tool
from search_tools import perplexity_search
perplexity_search_tool = FunctionTool(
    perplexity_search, 
    description="Search reputable inetrnet sites for recent news and up-to-date information using provided user_query, returns results with provided user query and search results"
)


#
# define junior agents
#

# ==================================================================================================
# Define a junior agent for economic_performance
# ==================================================================================================
economic_performance_agent_description = """
    an economic performance research agent that provides information on a country's economic performance through macroeconomic indicators
"""

economic_performance_system_prompt = """
    You are an experienced economic research analyst. 
    Your task is to assess the country's economic performance through macroeconomic indicators such as GDP growth, inflation rates, unemployment rates, and balance of payments. 
    Solve your task using tools to search for relevant, reliable and up to date information.
    This helps in understanding the overall health of the economy.
"""

economic_performance_agent = AssistantAgent(
    name="economic_performance_agent",
    tools=[perplexity_search_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    description=economic_performance_agent_description,
    system_message=economic_performance_system_prompt,
)

# ==================================================================================================
# Define a junior agent for Fiscal Analysis
# ==================================================================================================
fiscal_analysis_agent_description = """
    a Fiscal Analysis research agent that provides information on 
    the country government's fiscal policy, including its budget deficit or surplus, public debt levels, and the sustainability of its fiscal path.
"""

fiscal_analysis_system_prompt = """
    You are an experienced Fiscal Analysis research analyst. 
    Your task is to evaluate the government's fiscal policy, including its budget deficit or surplus, public debt levels, and the sustainability of its fiscal path.
    Review government budgets, debt levels, and fiscal policies to assess sustainability and default risk. 
    Solve your task using tools to search for relevant, reliable and up to date information.
    This helps in understanding the overall health of the country's financial situation and economy.
"""

fiscal_analysis_agent = AssistantAgent(
    name="fiscal_analysis_agent",
    tools=[perplexity_search_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    description=fiscal_analysis_agent_description,
    system_message=fiscal_analysis_system_prompt,
)

# ==================================================================================================
# Define a junior agent for Monetary Policy
# ==================================================================================================
monetary_policy_agent_description = """
    a Monetary Policy research agent that provides information on 
    the country central bank's monetary policy, including interest rate decisions, inflation targeting, and currency stability.
"""

monetary_policy_system_prompt = """
    You are an experienced Monetary Policy research analyst. Your task is to 
    analyze the central bank's monetary policy, including interest rate decisions, inflation targeting, and currency stability. 
    Perform External Sector Analysis to examine the current account balance, external debt levels, foreign exchange reserves, and trade dynamics.
    An independent and effective central bank can mitigate sovereign risk.
    Solve your task using tools to search for relevant, reliable and up to date information.
    This helps in understanding the overall health of the country's financial situation and economy.
"""

monetary_policy_agent = AssistantAgent(
    name="monetary_policy_agent",
    tools=[perplexity_search_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    description=monetary_policy_agent_description,
    system_message=monetary_policy_system_prompt,
)

# ==================================================================================================
# Define a senior agent for Report Generation
# ==================================================================================================

report_generation_agent_description = """
    a senior economic analyst agent that synthesizes information from junior agents to generate a comprehensive economic analysis report including executive summary.
"""

report_generation_system_prompt = """
    You are a senior economic analyst for a large fixed income investment company. 
    Your task is to synthesize information from junior agents on economic performance, fiscal analysis, and monetary policy to generate a comprehensive Sovereign risk analysis report. 
    
    Analyze the country's economic performance, fiscal policy, and monetary policy to provide a comprehensive assessment of the country's economic health and outlook.
    Review the macroeconomic indicators, fiscal policy, and monetary policy to identify key trends, risks, and opportunities.

    For Emergency Markets countries take into the coount that those countries are often characterized by more volatile economic growth. 
    Therefore, analyze past economic performance and future growth prospects . High growth rates can improve debt sustainability, but volatility can increase risks.

    Create detailed and comprehensive Sovereign Risk Analysis report that includes key findings, analysis, executive summary for policymakers, investors, and stakeholders.
    Check the report for accuracy, coherence, and relevance to the country's current economic situation. 
    Ask other agents for additional information if needed and ensure the report is coherent and well-researched.

    Respond 'TERMINATE' when task is complete.
"""

report_generation_agent = AssistantAgent(
    name="report_generation_agent",
    tools=[perplexity_search_tool],
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    description=report_generation_agent_description,
    system_message=report_generation_system_prompt,
)

# ==================================================================================================
# Define a group chat for economic analysis
# ==================================================================================================

economic_analysis_team = RoundRobinGroupChat(
    participants=[economic_performance_agent, fiscal_analysis_agent, monetary_policy_agent, report_generation_agent],
    termination_condition=TextMentionTermination("TERMINATE"),
    max_turns=15,
)

async def main():

    report_task = """
        Create Sovereign risk analysis report for Argentina. 
    """
    
    result_stream = economic_analysis_team.run_stream(task=report_task)
    async for message in result_stream:
            print(f'**> {message} \n\n')

    """
    await Console(
        economic_analysis_team.run_stream(
            task=report_task,
        )
    )
    """


import asyncio
asyncio.run(main())