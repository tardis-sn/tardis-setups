import requests
import os


def download_from_url(url):
    try:
        r = requests.get(url)
        fname = r.headers.get("Content-Disposit").split("filename=")[1]
    except AttributeError:
        fname = url.split("/")[-1].split("?")[0]

    file_path = os.path.join(os.path.expanduser("~"), "Downloads", "tardis-data", fname)
    open(file_path, "wb").write(r.content)
