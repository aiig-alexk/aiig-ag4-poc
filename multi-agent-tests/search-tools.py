# simple test to mimic risk analytic job. 
# the analyst agent uses other agents (data, analytic, news, etc.) to complete the report
# first iteration , copied from autogen LiteratureReview sample: https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/examples/literature-review.html


from autogen_agentchat.agents import CodingAssistantAgent, ToolUseAssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.components.tools import FunctionTool
from autogen_ext.models import OpenAIChatCompletionClient


# define functions and tools

import os
from openai import OpenAI
from typing import Optional

PERPLEXITY_API_KEY = "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"

ak_messages = [
    {
        "role": "system",
        "content": (
            "You are an expert finance, economic and political research analyst."
            "Provide comprehensive, up-to-date, and fact-based information with cited sources in response to the user queries"
            "Ensure all information is:"
            " - Sourced from reputable international news and economic reports"
            " - Dated within the last 3-6 months"
            " - Objective and factual"
        ),
    },
    {   
        "role": "user",
        "content": (
            "Ensure all information is:"
            " - Sourced from reputable international news and economic reports"
            " - Dated within the last 3-6 months"
            " - Objective and factual"
        ),
    },
]


def perplexity_search(country: str, user_query: Optional[str] = None) -> list:
    """
    Perform a search on Perplexity API to retrieve economic and political information about a specific country.
    
    Args:
        country (str): The name of the country to search for
        user_query (Optional[str]): Additional specific query about the country's economics or politics
    
    Returns:
        str: Comprehensive search results from Perplexity API
    """
    # Retrieve API key from environment variable
    # api_key = os.getenv('PERPLEXITY_API_KEY')
    api_key = PERPLEXITY_API_KEY
    if not api_key:
        raise ValueError("Perplexity API key not found. Set PERPLEXITY_API_KEY environment variable.")

    # Construct a comprehensive query
    default_query = (
        f"Provide a detailed and current analysis of {country}'s economic and political landscape. "
        "Include recent GDP, economic indicators, major industries, government structure, "
        "political developments, and international relations. "
        "Ensure information is up-to-date and sourced from credible international reports."
    )
    
    # Combine or use specific user query if provided
    # full_query = user_query or default_query
    full_query = f"Regarding {country}: {user_query}. Ensure the information is current and sourced from credible international reports."

    # Prepare messages for API call
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert economic and political research analyst specializing in providing detailed, "
                "up-to-date information about countries. Your responses should be "
                "comprehensive, fact-based, and include relevant context and sources."
                "Focus on verified, recent information from reputable international sources."
            )
        },
        {
            "role": "user", 
            "content": full_query
        }
    ]

    # Initialize Perplexity API client
    try:
        client = OpenAI(
            api_key=api_key, 
            base_url="https://api.perplexity.ai"
        )

        # Perform API call
        response = client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",  # Using the latest online model
            messages=messages,
            max_tokens=4096  # Adjust as needed
        )

        # Extract and return the response content
        enriched_results = []
        enriched_results.append(
            {"query": user_query, "results": response.choices[0].message.content.strip()}
        )
        
        return enriched_results

    except Exception as e:
        return f"Error retrieving information: {str(e)}"


# wrap the function into the tool
perplexity_search_tool = FunctionTool(
    perplexity_search, description="Search reputable inetrnet sites for recent news and up-to-date information, returns results with provided user quesry and search results"
)


# Example usage
if __name__ == "__main__":
   
    print("\n" + "="*50 + "\n")

    # Specific query: Argentina GDP
    specific_result = perplexity_search("Japan", "What are the current GNI (Gross National Income) and Foreign Direct Investment (FDI) Inflows")
    print(specific_result)

    print("\n" + "="*50 + "\n")
   

"""
Please help me to generate a python function "preplexity_search" that call perplexity API to get the latest information about some country economical and political facts and information based on user query. 
The function should take two arguments: "country" and "user query" and return the search result as text . 

For Perplexity API reference, use the following code sample:
<code>
from openai import OpenAI YOUR_API_KEY = "INSERT API KEY HERE" messages = [ { "role": "system", "content": ( "You are an artificial intelligence assistant and you need to " "engage in a helpful, detailed, polite conversation with a user." ), }, { "role": "user", "content": ( "How many stars are in the universe?" ), }, ] client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai") # chat completion without streaming response = client.chat.completions.create( model="llama-3.1-sonar-large-128k-online", messages=messages, ) print(response) # chat completion with streaming response_stream = client.chat.completions.create( model="llama-3.1-sonar-large-128k-online", messages=messages, stream=True, ) for response in response_stream: print(response)
</code>

Please adjust the code and prompts to use the argument "country" to provide the country context for the user query

"""




