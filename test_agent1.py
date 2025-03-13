#! custom_tools_example.py
import os
from typing import Optional

import litellm
from dotenv import load_dotenv
from smolagents import CodeAgent, Model, LiteLLMModel, tool, ToolCallingAgent

from bm25Tool.BM25RetrieverTool import BM25Tool
from converter.Converter import Converter

# Import your custom tools (replace with your actual tool imports)
# Assuming you have a file named 'custom_tools.py'
# and inside it, you have a BM25Tool class and other tools.

# LOAD THE ENV VARIABLES
load_dotenv(".env-local")

# GET GEMINI KEY FROM ENV FILE
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY2")

# set os environment variable GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# set litellm api key variable
LITELLM_API_KEY = "sk-".join(GEMINI_API_KEY)
litellm.api_key = LITELLM_API_KEY

model_ids = ["gemini/gemini-2.0-flash", "meta-llama/Llama-3.3-70B-Instruct", "anthropic/claude-3-5-sonnet-latest"]
model_id = 0

model: Model = LiteLLMModel(model_ids[model_id])

# Initialize custom tools
bm25_tool = BM25Tool()  # Assuming BM25Tool can be initialized without arguments
converter = Converter() #Assuming converter can be initialized without arguments

# Example 1: Using BM25Tool
bm25_query = "Summarize the text"
bm25_result = bm25_tool.forward(query=bm25_query)
print("BM25 Tool Result:")
print(bm25_result)
print("-" * 40)

# Example 2: Using Converter (assuming you have a test file 'test.txt')
test_file_path = "test.txt"  # Replace with your actual file path
with open(test_file_path, "w") as f:
    f.write("This is a test document.\nIt has multiple lines.\nAnd some words.")

converter.input_path = test_file_path
converter_result = converter.convert_txt_to_markdown()
print("Converter Tool Result:")
print(converter_result)
print("-" * 40)
#
# # Example 3: Integrating custom tools into a ToolCallingAgent
# @tool
# def process_document(file_path: str, query: str) -> str:
#     """
#     Processes a document using the converter and BM25 tools.
#     """
#     converter.input_path = file_path
#     markdown_content = converter.convert_txt_to_markdown()
#
#     # Assuming you save the converted markdown to a temporary file for BM25Tool
#     temp_markdown_file = "temp_converted.md"
#     with open(temp_markdown_file, "w") as f:
#         f.write(markdown_content)
#
#     # Assuming BM25Tool can use the temp file to create its index
#     bm25_result = bm25_tool.forward(query=query)
#
#     # Clean up temporary file
#     os.remove(temp_markdown_file)
#
#     return bm25_result
#
# tool_agent = ToolCallingAgent(
#     tools=[process_document, bm25_tool, converter],  # Add your custom tools
#     model=model,
#     add_base_tools=True,
#     verbosity_level=2,
# )
#
# document_processing_query = f"Process the document {test_file_path} and summarize it."
# tool_agent_result = tool_agent.run(document_processing_query)
# print("Tool Calling Agent Result:")
# print(tool_agent_result)
# print("-" * 40)
#
# # Example 4: Integrating custom tools into a CodeAgent
# code_agent = CodeAgent(
#     tools=[bm25_tool, converter],
#     model=model,
#     add_base_tools=True,
#     additional_authorized_imports=["os", "pathlib"],
#     verbosity_level=2,
# )
#
# code_agent_task = f"""
# 1. Use the Converter tool to convert the file 'test.txt' to markdown.
# 2. Use the BM25 tool to summarize the converted markdown content.
# 3. Print the result.
# """
# code_agent.run(code_agent_task)