from gdown import download, download_folder
from requests import get
from os.path import join, exists
from zipfile import ZipFile
from os import makedirs


def download_file_drive(url: str, output: str, quiet: bool = False) -> None:
    if exists(output):
        print(f"{output} already exists")
        return

    download(url=url, output=output, quiet=quiet, fuzzy=True)


def download_folder_drive(
    url: str, output: str, quiet: bool = False, use_cookies: bool = False
) -> None:
    if exists(output):
        print(f"{output} already exists")
        return

    download_folder(url=url, output=output, quiet=quiet, use_cookies=use_cookies)


def download_file(url: str, output: str, parent: str = None):
    if parent:
        makedirs(parent, exist_ok=True)

    path = output if parent is None else join(parent, output)

    if exists(path):
        print(f"{path} is already exists")
        return

    request = get(url)
    print(f"Downloading {output} to {path}")

    with open(path, "wb") as file:
        file.write(request)


def download_zip(url: str, output: str, parent: str = None):
    if parent:
        makedirs(parent, exist_ok=True)

    temp_path = join("temp", output)
    real_path = output if parent is None else join(parent, output)

    if exists(real_path):
        print(f"{real_path} is already exists")
        return

    request = get(url)
    print(f"Downloading {output}")

    with open(temp_path, "wb") as file:
        file.write(request)

    with ZipFile(temp_path, "r") as zipfile:
        print(f"Unzipping {output} to {real_path}")
        zipfile.extractall(real_path)
