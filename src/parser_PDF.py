import pymupdf4llm


class ParserPDF:
    def __init__(self):
        pass

    def read_markdown_from_file(self, filepath: str) -> str:
        return pymupdf4llm.to_markdown(filepath)

    # def clean_text(self, text: str) -> str:



if __name__ == "__main__":
    parser = ParserPDF()
    text = parser.read_markdown_from_file("test/test_document.pdf")
    print(text[:10000])





