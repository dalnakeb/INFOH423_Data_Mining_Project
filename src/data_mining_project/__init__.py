import os as _os
from pathlib import Path


PROJECT_PATH = Path(_os.getcwd()).parent.absolute()
DATA_PATH = PROJECT_PATH / 'data'
OUTPUT_PATH = PROJECT_PATH / 'outputs'