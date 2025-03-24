import re

from config import OUTPUT_DIRECTORY

output_directory = OUTPUT_DIRECTORY

def generate_toc(md_content: str) -> str:
    """
    Generates a table of content from a Markdown document
    :param md_content: The content of the markdown file.
    :return: The computed table of content.
    """
    toc_lines = []
    for line in md_content.splitlines():
        match = re.match(r"^([#|*|$]+)\s*(.*)", line)
        if match and len(match.group(1)) > 1 and len(match.group(2).strip()) >= 5:
            level: int = len(match.group(1))
            title: str = match.group(2).strip()
            desired_level: int = level-2
            indent: str = "    " + str(desired_level)
            toc_lines.append("{} - {}".format(indent, title))
    return "\n".join(toc_lines)
