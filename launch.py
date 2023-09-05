from VenvManager import VenvManager

if __name__ == "__main__":
    manager = VenvManager()

    manager.InstallWRequirements()

    manager.RunScript("main")
