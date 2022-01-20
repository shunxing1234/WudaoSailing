# 
import copy
import fnmatch
import functools
import importlib.util
import io
import json
import os
import re
import shutil
import sys
from functools import partial
import tarfile
import tempfile
import types
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO, ContextManager, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from uuid import uuid4
from zipfile import ZipFile, is_zipfile
import numpy as np
from packaging import version
from tqdm.auto import tqdm
import requests
from filelock import FileLock
import logging


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})
SESSION_ID = uuid4().hex
logger = logging.getLogger(__name__.split(".")[0]) 


class DataPull(object):
    def __init__(self):
        self.models_url={
                'glm-10b-en':'https://dorc.baai.ac.cn/resources/MDHub/glm-10b-en.tar.gz',
                'glm-10b-zh':'https://dorc.baai.ac.cn/resources/MDHub/glm-10b-zh.tar.gz',
                'glm-large-en-blank':'https://dorc.baai.ac.cn/resources/MDHub/glm-large-en-blank.tar.gz',
                'glm-large-zh':'https://dorc.baai.ac.cn/resources/MDHub/glm-large-zh.tar.gz'
                 }
        self.zip_from_url=zip_from_url
        self.download_from_url=download_from_url
    def pull_model(self,model_name=None):
        res = requests.get("https://dorc.baai.ac.cn/api/hub/v1/generateTempLink?name=" +model_name )
        res.encoding = "utf-8"
        urls=json.loads(res.text)['data']

        for url in urls:
            #datadown.download_from_url(datadown.models_url['glm-large-zh'],file_pname=model_name+'.tar.gz')
            self.zip_from_url(self.models_url['glm-large-zh'],output_path_extracted='./checkpoints/checkpoints')

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def url_to_hash(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()
    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()
    if url.endswith(".h5"):
        filename += ".h5"
    return filename

def remove_dir_with_sub_dirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if os.path.exists(path):
        os.rmdir(path)

def download_from_url(url, output_path_extracted='./checkpoints/',file_pname=None, chunk_size=1024*256,resume_download=True):
    """
    url: file url
    file_pname: file save name
    chunk_size: chunk size
    resume_download: download from last chunk
    """
    response = requests.get(url, stream=True, verify=True)  
    try:      
        total_size = int(response.headers['Content-Length'])  
    except Exception as e: 
        raise ValueError('please check the url')      
    
    if file_pname==None:
        file_path=output_path_extracted+url.split('/')[-1]
    if not os.path.exists(output_path_extracted):
        os.mkdir(output_path_extracted)
    if os.path.exists(file_path):  
        resume_size = os.path.getsize(file_path)  
    else:       
        resume_size = 0  
    headers = {'Range': 'bytes=%d-' % resume_size}   
    res = requests.get(url, stream=True, verify=True, headers=headers)  
    progress = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=total_size,
            initial=resume_size,
            desc="Downloading",
            disable=bool(logging.getLogger(__name__.split(".")[0]) == logging.NOTSET),
        )
    while 1:
        with open(file_path, "ab") as f:  
            for chunk in res.iter_content(chunk_size=1024*1024):   
                if chunk:       
                    f.write(chunk) 
                    progress.update(len(chunk))
                    f.flush()   
                    
        resume_size = os.path.getsize(file_path)  
        if resume_size>=total_size:
            print('model dowloaded in ',os.getcwd()+output_path_extracted[1:])
            break
        else:
            headers = {'Range': 'bytes=%d-' % resume_size}   
            res = requests.get(url, stream=True, verify=True, headers=headers)             


def cache_load(
    url: str,
    cache_dir=None,
    resume_download=True
) -> Optional[str]:
 
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = './cache_files/'        
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    response = requests.get(url, stream=True, verify=True)  
    try:      
        total_size = int(response.headers['Content-Length'])  
    except Exception as e: 
        raise ValueError('please check the url') 
        
    filename = url_to_hash(url)
    cache_path = os.path.join(cache_dir, filename)
    @contextmanager
    def _resumable_file_manager() -> "io.BufferedWriter":
        with open(cache_path, "ab") as f:
            yield f
    if resume_download:
        temp_file_manager = _resumable_file_manager
        if os.path.exists(cache_path): 
            resume_size = os.path.getsize(cache_path)
            if resume_size>=total_size:
                print('cache  saved in ',cache_path)
                return cache_path
        else:       
            resume_size = 0
    else:
        temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
        resume_size = 0
    progress = tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=total_size,
        initial=resume_size,
        desc="Downloading",
        disable=bool(logging.getLogger(__name__.split(".")[0]) == logging.NOTSET),
    )  
    while 1:
        headers = {'Range': 'bytes=%d-' % resume_size}
        res = requests.get(url, stream=True, verify=True, headers=headers)
        with temp_file_manager() as temp_file:
            for chunk in res.iter_content(chunk_size=1024*256):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
        resume_size = os.path.getsize(cache_path)
        if resume_size>=total_size:
            print('cache  saved in ',cache_path)
            return cache_path
        temp_file_manager = _resumable_file_manager
    progress.close() 
    print('cache  saved in ',cache_path)
    return cache_path


def extract_chche_files(cache_path,output_path_extracted):
    if  output_path_extracted is None:
        output_path_extracted ='./checkpoints/'
    if not is_zipfile(cache_path) and not tarfile.is_tarfile(cache_path):
        return cache_path
    # Path where we extract compressed archives
    # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
    if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted):
        return output_path_extracted
    # Prevent parallel extractions
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):
        shutil.rmtree(output_path_extracted, ignore_errors=True)
        if not os.path.exists(output_path_extracted):os.makedirs(output_path_extracted)
        if is_zipfile(cache_path):
            with ZipFile(cache_path, "r") as zip_file:
                zip_file.extractall(output_path_extracted)
                zip_file.close()
        elif tarfile.is_tarfile(cache_path):
            tar_file = tarfile.open(cache_path)
            tar_file.extractall(output_path_extracted)
            tar_file.close()
        else:
            raise EnvironmentError(f"Archive format of {output_path} could not be identified")
    return cache_path


def zip_from_url(
    url,
    cache_dir=None,
    resume_download=True,
    extract_compressed_file=True,
    output_path_extracted=None
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = './cache_files'
    if isinstance(url, Path):
        url_or_filename = str(url)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    # URL, so get it from the cache (downloading if necessary)
    cache_path = cache_load(
        url,
        cache_dir=cache_dir,
        resume_download=resume_download
    )
    if extract_compressed_file:
        output_path=extract_chche_files(cache_path,output_path_extracted)
        print('model downloaded in ',output_path_extracted)
    return output_path
