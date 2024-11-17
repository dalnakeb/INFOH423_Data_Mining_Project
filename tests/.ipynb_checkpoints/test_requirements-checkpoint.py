import importlib.metadata
from pathlib import Path


def test_verify_requirements():
    """
    Verify it the packages and requirements are respected as noted in requirements.txt
    :return: None
    """
    required_packages = {}
    file_path = Path(__file__).parent.parent.absolute() / "requirements.txt"
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                package, version = line.split('==')
                required_packages[package] = version

    for package, required_version in required_packages.items():
        installed_version = importlib.metadata.version(package)
        p1 = str(package) + str(installed_version)
        p2 = str(package) + str(required_version)
        assert p1 == p2
