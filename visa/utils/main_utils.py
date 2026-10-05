import os
import sys

import numpy as np
import dill
import yaml
from pandas import  DataFrame

from visa.exception import USvisaException
from visa.logger import logging

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise USvisaException(e,sys) from e

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.join(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            yaml.dump(content,file)
    except Exception as e:
        raise USvisaException(e,sys) from e

def load_object(file_path:str)->object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path,'rb') as file_obj:
            obj=dill.load(file_path)
        logging.info("Excitted the load_object method of utils")
        return obj
    except Exception as e:
        raise USvisaException(e,sys) from e

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise USvisaException(e,sys) from e
    
def load_numpy_array_data(file_path:str)->np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise USvisaException(e,sys) from e
    
def save_object(file_path:str,obj:object)->None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
        logging.info("Exited the save_object method of utils") 
    except Exception as e:
        raise USvisaException(e,sys)  from e
     

def drop_columns(df:DataFrame,cols:list)->DataFrame:
    logging.info("Entered drop_columns method of utils")
    try:
        df=df.drop(columns=cols,axis=1)
        
        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise USvisaException(e,sys) from e