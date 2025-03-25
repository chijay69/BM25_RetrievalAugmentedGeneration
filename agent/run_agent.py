"""run_agent.py"""

import os

import litellm
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

from agent.bm25 import BM25Tool
from config_reader import get_base_directory

# Initialize custom tools
bm25_tool: BM25Tool = BM25Tool()  # Assuming BM25Tool can be initialized without arguments

env_path: str = os.path.join(get_base_directory(), ".env-local")
# LOAD THE ENV VARIABLES
load_dotenv(env_path)

# GET GEMINI KEY FROM ENV FILE
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY2")

os.putenv("GEMINI_API_KEY", GEMINI_API_KEY)


# set litellm api key variable
litellm.api_key = GEMINI_API_KEY

model_ids = ["gemini/gemini-2.0-flash", "meta-llama/Llama-3.3-70B-Instruct", "anthropic/claude-3-5-sonnet-latest"]
model_id = 0

model: LiteLLMModel = LiteLLMModel(model_ids[model_id])

agent: CodeAgent = CodeAgent(
    tools= [bm25_tool],
    model= model,
    add_base_tools= False,
    additional_authorized_imports=["os", "asyncio"],
    verbosity_level= 2,
)

result = agent.run("List the advantages of TM30 platform")

print("Agent's response: \t", result)