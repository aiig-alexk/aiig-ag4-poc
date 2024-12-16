#
# Economic Data Search Tools
#
import os
import json
import logging

from openai import OpenAI
from typing import Optional

logger = logging.getLogger(__name__)

def perplexity_search(country: str, user_query: Optional[str] = None) -> list:
    """
    Perform a search on Perplexity API (https://docs.perplexity.ai) to retrieve recent economic and political information about a specific country.
    
    Args:
        country (str): The name of the country to search for
        user_query (Optional[str]): Additional specific query about the country's economics or politics
    
    Returns:
        str: Comprehensive search results from Perplexity API
    """

    logging.info(f"===> perplexity_search call: {country} - {user_query}")

    # Retrieve API key from environment variable
    pplx_api_key = os.getenv('PERPLEXITY_API_KEY')
    if not pplx_api_key:
        logging.error("Perplexity API key not found. Set PERPLEXITY_API_KEY environment variable.")
        raise ValueError("Perplexity API key not found. Set PERPLEXITY_API_KEY environment variable.")

    pplx_model = "llama-3.1-sonar-large-128k-online" # Using the latest online model

    # Detailed system prompt. @todo: Consider moving some parts to the user message
    pplx_search_system_prompt = """
        You are an expert economic and political research AI Assitant specializing in providing detailed up-to-date information about countries. 
        Your responses are accurate, comprehensive, fact-based, and include relevant indicators, statistics and quantitative analytics. 
        Focus on verified, recent information from reputable and credible sources such as: 
            - World Bank World Development Indicators (WDI): A comprehensive database of development indicators, including GDP growth, inflation, and trade statistics.
            - International Monetary Fund (IMF) World Economic Outlook (WEO): Provides economic analysis and projections for countries.
            - United Nations Statistics Division: Offers a range of economic indicators and statistics.
            - CIA World Factbook: CIA World Factbook: Provides basic intelligence on the history, people, government, economy, energy, geography, environment, communications, transportation, military, terrorism, and transnational issues for 265 world entities.
            - CEIC (www.ceicdata.com): Curates the best and most relevant economic, industry and financial data for economists and investment professionals to track and gain genuine insight into what is happening in their markets – with a particular focus on emerging economies.
        Include the list of URLs for all citations, references and sources used in the response.
        Ensure all responses are accurate, factual and supported by authoritative sources. If exact data is unavailable, clarify and provide context or relevant approximations.
    """

    # alternative system prompt by ChatGPT
    pplx_search_system_prompt_gpt = """
        You are an expert AI assistant specializing in providing concise, accurate, and up-to-date information 
        about countries' financial data, economic conditions, political situations, and key indicators such as GDP, inflation, unemployment rate, government stability, and major trade agreements. 
        Ensure all responses are factual and supported by authoritative sources. If exact data is unavailable, clarify and provide context or relevant approximations."
    """

    # Combine country and user quesry 
    full_query = f"Regarding {country}: {user_query}."
 
    # Prepare messages for API call. Always start from scratch and do not include previous replies from "assistant"
    messages = [
        {
            "role": "system", 
            "content": pplx_search_system_prompt
        },
        {
            "role": "user", 
            "content": full_query
        }
    ]

    logging.info(f"===| perplexity_search: Full query: {full_query} - System Message: {messages}")

    # Initialize Perplexity API client
    try:
        client = OpenAI(
            api_key=pplx_api_key, 
            base_url="https://api.perplexity.ai"
        )

        # Perform API call
        response = client.chat.completions.create(
            model=pplx_model,  
            messages=messages,
            max_tokens=4096,  # Adjust as needed
            # @todo: consider using "search_recency_filter" : Returns search results within the specified time interval
        )

        logging.info(f"===| perplexity_search full response: {response}")     # @todo: consider adding user_query to the response

        # Extract and return the response content
        enriched_results = []
        enriched_results.append({
            "query": user_query, 
            "results": response.choices[0].message.content.strip(),
            "citations": response.citations
        })
        
        logging.info(f"===< perplexity_search results: {enriched_results}")

        return enriched_results

    except Exception as e:
        logging.error(f"Error retrieving information: {str(e)}")
        return f"Error retrieving information: {str(e)}"


# ==================================================================================================
# Example usage
# ==================================================================================================
if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
   
    print("\n" + "="*50 + "\n")

    # Specific query: Argentina GDP
 #   specific_result = perplexity_search("Argentina", "What are the current GNI (Gross National Income) and Foreign Direct Investment (FDI) Inflows")
    specific_result = perplexity_search("Argentina", "current budget deficit in 2024")
 
    formatted_result = json.dumps(specific_result, indent=4)
    print(f"****>>> formatted results: {formatted_result}")

    print("\n" + "="*50 + "\n")
   

"""
Please help me to generate a python function "preplexity_search" that call perplexity API to get the latest information about some country economical and political facts and information based on user query. 
The function should take two arguments: "country" and "user query" and return the search result as text . 
"""