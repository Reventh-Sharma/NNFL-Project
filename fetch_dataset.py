import os
import requests
import zipfile
from io import BytesIO

# COIL-100 Dataset URL
DATASET_URL = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
# Dataset Directory ('./dataset')
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')

if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

response = requests.get(DATASET_URL, stream=True)

if response.status_code != 200:
    raise ValueError(f"GET request returned {response.status_code}. Please check the URL or try again later.")

zip_file = zipfile.ZipFile(BytesIO(response.content))
zip_file.extractall(DATASET_DIR)