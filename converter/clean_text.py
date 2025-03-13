import string


# A text cleaning function
def clean_text(text: str)->str:
    """
    Cleans the text argument passed
    :param text: A text.
    :return: A text.
    """
    lower_case_text: str = text.lower()
    translated_text: str = lower_case_text.translate(str.maketrans("","",string.punctuation))
    # TODO
    # Do stemming operation
    return translated_text

