import asyncio
import os
import os.path
from logging import Logger

from bm25Tool.build_document_index import read_file_content
from bm25Tool.gen_toc import generate_toc
from bm25Tool.setup_logger import setup_logger
from config_reader import get_base_directory, get_output_dir
from converter.SaveFile import save_file_to_path

BASE_DIR: str = get_base_directory()
CONFIG_PATH: str = os.path.join(BASE_DIR, "config.ini")
OUTPUT_DIRECTORY: str = os.path.join(BASE_DIR, get_output_dir(CONFIG_PATH))


def create_toc(output_path: str) -> None:
    """
    Generates and saves table of content files for each file in the specified directory.

    :param output_path: The directory containing the files to process.
    """

    logger: Logger = setup_logger(__name__)
    logger.info(f"Starting table of contents generation in: {output_path}")

    try:
        if not os.path.exists(output_path):
            logger.error(f"Directory not found: {output_path}")
            print(f"Error: Directory not found: {output_path}")
            return

        for filename in os.listdir(output_path):
            filepath = os.path.join(output_path, filename)

            if os.path.isfile(filepath) and not filename.endswith("_toc.md"):  # Process only files and skip existing TOCs
                logger.info(f"Processing file: {filename}")
                file_content: str = read_file_content(filepath)
                toc_content: str = generate_toc(file_content)

                toc_filename = os.path.splitext(filename)[0] + "_toc.md"
                toc_filepath = os.path.join(output_path, toc_filename)

                save_file_to_path(toc_content, toc_filepath)
                logger.info(f"Table of contents saved to: {toc_filepath}")
                print(f"Table of contents saved to: {toc_filepath}")

        logger.info("Table of contents generation completed.")

    except (FileNotFoundError, OSError, Exception) as e:
        logger.error(f"An error occurred during TOC generation: {e}")
        print(f"An error occurred: {e}")

