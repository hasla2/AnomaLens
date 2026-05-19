# pip install requests beautifulsoup4

# wget -r -l 2 --no-parent -A "*.html" \
#  "https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/" \
#  -P data/docs/hyperv/

import os
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_html(url, output_dir, visited, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        return

    if url in visited:
        return

    visited.add(url)

    try:
        print(f"[+] Downloading: {url}")

        response = requests.get(url, headers=HEADERS, timeout=30)

        if response.status_code != 200:
            print(f"[!] Failed: {url} ({response.status_code})")
            return

        if "text/html" not in response.headers.get("Content-Type", ""):
            return

        parsed = urlparse(url)

        # Creating file path
        path = parsed.path.strip("/")

        if not path:
            path = "index"

        filename = os.path.join(output_dir, path)

        if not filename.endswith(".html"):
            filename += ".html"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"[✓] Saved: {filename}")

        # Parcing links
        soup = BeautifulSoup(response.text, "html.parser")

        for link in soup.find_all("a", href=True):
            href = link["href"]

            full_url = urljoin(url, href)

            # Only that same domain
            if urlparse(full_url).netloc != parsed.netloc:
                continue

            # Only HTML pages
            if any(
                full_url.endswith(ext)
                for ext in [".jpg", ".png", ".pdf", ".zip", ".exe"]
            ):
                continue

            download_html(full_url, output_dir, visited, max_depth, current_depth + 1)

    except Exception as e:
        print(f"[ERROR] {url}: {e}")


if __name__ == "__main__":

    targets = [
        {
            "url": "https://knowledge.broadcom.com/external/article/344837/main-kb-list-of-vsphere-80-knowledge-ba.html",
            "dir": "data/docs/vmware",
        },
        {
            "url": "https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/",
            "dir": "data/docs/hyperv",
        },
        {
            "url": "https://learn.microsoft.com/en-us/troubleshoot/windows-server/welcome-windows-server",
            "dir": "data/docs/hyperv",
        },
        {
            "url": "https://techdocs.broadcom.com/us/en/vmware-cis/vsphere/vsphere/8-0.html",
            "dir": "data/docs/vmware",
        },
        {"url": "https://www.reddit.com/r/vmware/", "dir": "data/docs/vmware"},
    ]

    for target in targets:
        visited_urls = set()

        download_html(
            url=target["url"],
            output_dir=target["dir"],
            visited=visited_urls,
            max_depth=2,
        )
