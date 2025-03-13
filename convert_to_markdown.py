import json
import os

import pymupdf4llm
from converter.Converter import Converter

from config import INPUT_DIRECTORY, OUTPUT_DIRECTORY


# load configuration files
with open("config.json", "r") as config_file:
    config = json.loads(config_file.read())

section_size = config["section_size"]

# Define input and output directories
input_directory = INPUT_DIRECTORY
output_directory = OUTPUT_DIRECTORY

# create output directory if it does not exist
os.makedirs(output_directory, exist_ok=True)

converter = Converter()

# process each file in the input directory
for filename in os.listdir(input_directory):
    input_path: str =  os.path.join(input_directory, filename)

    if filename.endswith(".pdf"):
        # convert pdf to markdown
        md_text: str = pymupdf4llm.to_markdown(input_path)
        # Remove page delineators (common patterns are '------------' or page numbers)
        md_text = "\n".join(line for line in md_text.split("\n") if not line.strip().startswith("------") and not (line.strip().isdigit() or line.strip() == " "))
    elif filename.endswith(".docx"):
        # Convert docx to markdown
        md_text = converter.convert_doc_to_markdown(input_path)
    elif filename.endswith(".doc"):
        # Convert doc to markdown
        md_text = converter.convert_doc_to_markdown(input_path)
    else:
        continue


