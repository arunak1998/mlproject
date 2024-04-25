import os
import sys

import pandas
import dill
import numpy
from src.exception import Customexception
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
def save_obj(file_path,obj):

    try:
        dir_path=os.path.dirname(file_path)
        

        os.makedirs(dir_path,exist_ok=True)

        

# Open the file for writing in binary mode ('wb') and dump the object using dill
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
       raise Customexception(e,sys)


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        try:
     
            X_outliers_removed = X.copy()
            upper_limit = X[self.col].mean() + 3 * X[self.col].std()
            lower_limit = X[self.col].mean() - 3 * X[self.col].std()
            X_outliers_removed = X_outliers_removed[(X_outliers_removed[self.col] > lower_limit) & (X_outliers_removed[self.col] < upper_limit)]
           
            return X_outliers_removed
        except Exception as e:
            raise Customexception(e,sys)
        


def evaluvate_model(X_train,X_test,y_train,y_test,model):
    try:
            report={}
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            # make predictions
            y_test_pred = model.predict(X_test)
            # Calculate the accuracy of the model

            accuracy_train = accuracy_score(y_train, y_train_pred)
            accuracy_test= accuracy_score(y_test, y_test_pred)
            conf_matrix_train= confusion_matrix(y_train, y_train_pred)
            conf_matrix_test= confusion_matrix(y_test, y_test_pred)
            
            report[model]=accuracy_test
            return report
            
    except Exception as e:

            raise Customexception(e,sys)