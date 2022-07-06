import requests
import os


def download_from_url(url):
    r = requests.get(url)
    fname = r.headers.get("Content-Disposition").split("filename=")[1]
    file_path = os.path.join(os.path.expanduser("~"), "Downloads", "tardis-data", fname)
    open(file_path, "wb").write(r.content)
