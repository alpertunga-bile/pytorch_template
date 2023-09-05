from VenvManager import VenvManager
from os import makedirs
from shutil import rmtree

if __name__ == "__main__":
    manager = VenvManager()

    manager.InstallWRequirements()

    makedirs("temp", exist_ok=True)

    manager.RunScript("main")

    rmtree("temp")
