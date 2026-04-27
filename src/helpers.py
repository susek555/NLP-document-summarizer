import re

import feedparser
import requests

from src.text_object_type import TextObjectType


def download_and_save_arxiv_pdf(arxiv_id: str, name: str):
    response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}.pdf", timeout=10)
    with open(f"{name}{TextObjectType.BASE_PDF.value}", "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {name}.pdf")


def download_arxiv_abstract(arxiv_id: str):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    feed = feedparser.parse(url)

    if not feed.entries:
        return ""

    abstract = feed.entries[0].summary
    abstract = abstract.replace("\n", " ")
    abstract = re.sub(r"\s+", " ", abstract).strip()

    return abstract


def save(name: str, type: TextObjectType, text: str) -> None:
    with open(f"{name}{type.value}", "w") as f:
        f.write(text)


def read(name: str, type: TextObjectType) -> str:
    with open(f"{name}{type.value}") as f:
        return f.read()


if __name__ == "__main__":
    download_and_save_arxiv_pdf("1706.03762", "test/test_document")
