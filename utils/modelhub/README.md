# modelhub

modelhub is a tool for quick start.


## Get Started

The [utils/modelhub/client.py](utils/modelhub/client.py) contains the API for downing models and files.

To down the model [glm-10b-en], run
  ```python
from utils.modelhub.client import DataPull       
if __name__=='__main__':
    datadown=DataPull()
    datadown.pull_model("glm-10b-en")         
  ```
  
also, one can download model by model_url as
  ```python
from client import zip_from_url
zip_from_url(url,to_path='./checkpoints/')      
  ```
[zip_from_url](utils/modelhub/client.py) support downing zip files and tar.gz files and unzip files to path "to_path", for other type files, please use
[download_from_url](utils/modelhub/client.py) as
  ```python
from client import download_from_url
download_from_url(url,to_path='./checkpoints/')      
  ```

 
