import os
from pathlib import Path

files_list = [
    f"notebooks/01_data_ingestion.ipynb",
    f"src/components/__init__.py",
    f"src/pipelines/__init__.py",
    f"src/__init__.py",
    f"src/utils.py",
    f"src/logger.py",
    f"requirements.txt"
]

for file_path in files_list:

    file_path = Path(file_path)

    file_dir, file_name = os.path.split(file_path)

    try:
        if file_dir!='':
            os.makedirs(file_dir, exist_ok=True)
    except:
        raise Exception
    
    try:
        if not os.path.exists(file_path) or not os.path.getsize(file_path)==0:
            with open(file_path, 'w'):
                pass
    except:
        raise Exception