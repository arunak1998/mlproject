import pandas as pd
import numpy as np
import os 
import sys
from dataclasses import dataclass
from src.exception import Customexception
from src.logger import logging
from xgboost import XGBClassifier

from src.utils import save_obj,evaluvate_model
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


@dataclass
class ModelTrainerConfig:
    preprocessor_obj_file=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) :
        self.model_train_config=ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging .info("Splliting Traning and Test data ")

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            )
          

            model = XGBClassifier(
                n_estimators=100, 
                objective='binary:logistic',  # Objective for binary classification
                gamma=0.1,                     # Minimum loss reduction required to make a further partition on a leaf node of the tree
                reg_lambda=1,                  # L2 regularization term on weights
                learning_rate=0.1              # Boosting learning rate (eta)
                )
            model_report:dict=evaluvate_model(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,model=model)

            value_list = list(model_report.values())
            model_score=value_list[0]
            print(model_score)

            if model_score<0.8:
                raise Customexception("This is not best model")
            logging.info("This is best model for this ")


            save_obj(
                file_path=self.model_train_config.preprocessor_obj_file,
                obj=model
            )

            predicted=model.predict(X_test)

            accuracy=accuracy_score(y_test,predicted)

            return accuracy
        
        except Exception as e:
            raise Customexception(e,sys)
