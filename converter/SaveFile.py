from logging import Logger

from bm25Tool.setup_logger import setup_logger


# function to save file to path

def save_file_to_path(content: str | bytes, filepath: str) -> None:
    """
    Saves the content to the specified file path.
    :param content: The content to save (string or bytes).
    :param filepath: The full path to the file.
    """
    logger: Logger = setup_logger(__file__)
    try:
        if isinstance(content, str):
            mode = "w"  # Write text
        elif isinstance(content, bytes):
            mode = "wb" # Write bytes
        else:
            msg: str = "Content must be a string or bytes."
            logger.error(msg)
            raise ValueError(msg)

        with open(filepath, mode, encoding="UTF-8") as f:
            f.write(content)
    except OSError as e:
        logger.error(f"Error saving file: {e}")
    except ValueError as e:
        logger.error(e)

