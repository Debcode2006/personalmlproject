import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


from src.exceptions import CustomException

def save_object(file_path, obj):
    '''
    This function saves the object in the specified file directory.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models):
    '''
    This function evaluates the models and returns the r2 scores.
    '''
    try:
        report = {}
        
        for i in range (len(list(models))):
            model = list(models.values())[i]
            
            model.fit(x_train, y_train)
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            train_r2_score = r2_score(y_train_pred, y_train)
            test_r2_score = r2_score(y_test_pred, y_test)
            
            report[list(models.keys())[i]] = test_r2_score
            
        return report
    
    
    
    except Exception as e:
        raise CustomException(e,sys)