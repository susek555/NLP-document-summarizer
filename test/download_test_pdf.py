import requests


def download_pdf(url, filename="test/test_document.pdf"):
    response = requests.get(url, timeout=10)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {filename}")

if __name__ == "__main__":
    download_pdf("https://arxiv.org/pdf/1706.03762.pdf")
