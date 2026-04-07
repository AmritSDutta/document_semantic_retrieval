import re


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)  # emails
    text = re.sub(r'\d+', ' ', text)  # numbers
    text = re.sub(r'www\S+', ' ', text)  # urls
    text = re.sub(r'\b[a-z]{1,2}\b', ' ', text)  # short junk
    return text
