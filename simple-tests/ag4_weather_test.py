import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models import OpenAIChatCompletionClient


# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."


async def main() -> None:
    # Define an agent
    my_api_key=os.getenv('OPENAI_API_KEY')
    if my_api_key is None:
        print("Environment variable 'OPENAI_API_KEY' is not set!")
    else:
        print(f"API Key: {my_api_key}")

    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
            api_key=my_api_key
        ),
        tools=[get_weather],
    )

    # Define termination condition
    termination = TextMentionTermination("TERMINATE")

    # Define a team
    agent_team = RoundRobinGroupChat([weather_agent], termination_condition=termination)

    # Run the team and stream messages to the console
    stream = agent_team.run_stream(task="What is the weather in New York?")
    await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
# await main()
asyncio.run(main())