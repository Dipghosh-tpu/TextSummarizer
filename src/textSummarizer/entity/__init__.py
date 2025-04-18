from dataclasses import dataclass
from pathlib import Path
import requests

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path
