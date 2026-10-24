import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

from langchain.tools import tool


os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is not set in the environment variables.")

if not WEATHERSTACK_API_KEY:
    raise ValueError("WEATHERSTACK_API_KEY is not set in the environment variables.")


search_tool = TavilySearch(
    max_results=2, api_key=TAVILY_API_KEY
)

@tool
def get_weather_data(location: str) -> str:
    """
    Get the current weather for a given location.
    """

    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={location}"
    response = requests.get(url)
    data = response.json()

    if "current" in data:
        temperature = data["current"]["temperature"]
        description = data["current"]["weather_descriptions"][0]
        return f"The current weather in {location} is {description} with a temperature of {temperature}°C."
    else:
        return f"Could not retrieve weather data for {location}."
    
    
tools = [search_tool, get_weather_data]


llm = ChatOpenAI(
    model_name="gpt-4o",
    api_key=OPENAI_API_KEY,
    temperature=0,
    max_retries=3,
    timeout=30,
    max_tokens=200
    
)


# Prompt for the agent
SYSTEM_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


agent = create_agent(
    model = llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# =========================================================
# 7. Run Agent with ReAct-Style Action Logs
# =========================================================

def run_agent(user_query: str) -> None:
    """Stream agent output and display actions and observations."""

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": user_query,
            }
        ]
    }

    print("\n" + "=" * 60)
    print(f"User: {user_query}")
    print("=" * 60)

    for chunk in agent.stream(
        inputs,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            messages = update.get("messages", [])

            for message in messages:

                # -----------------------------------------
                # Agent Action
                # -----------------------------------------
                if getattr(message, "tool_calls", None):
                    for action in message.tool_calls:
                        print("\nAction:")
                        print(f"  Tool: {action['name']}")
                        print(f"  Action Input: {action['args']}")

                # -----------------------------------------
                # Tool Observation
                # -----------------------------------------
                elif getattr(message, "type", "") == "tool":
                    print("\nObservation:")
                    print(message.content)

                # -----------------------------------------
                # Agent Final Answer
                # -----------------------------------------
                elif (
                    getattr(message, "type", "") == "ai"
                    and message.content
                    and not getattr(message, "tool_calls", None)
                ):
                    print("\nFinal Answer:")
                    print(message.content)

    print("\n" + "=" * 60)
    print("Agent execution completed.")
    print("=" * 60)


# =========================================================
# 8. Test Agent
# =========================================================

if __name__ == "__main__":
    run_agent("Give me the current weather in Dhaka, Bangladesh")