import sys
import pandas as pd
import numpy
from src.exception import Customexception
from src.logger import logging

from src.utils import load_object


class PredictData:
    def __init__(self) :
        pass
    def predict(self,features):

        try:

            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(features)
            print(data_scaled)
            pred=model.predict(data_scaled)

            print(pred)

            return pred
        except Exception as e:
            raise Customexception(e,sys)

class CustomerData:
    def __init__(self, CreditScore: int, Age: int, Tenure: int, Balance: float, NumOfProducts: int, HasCrCard: int, IsActiveMember: int, EstimatedSalary: float, Gender: str, Geography: str):
        self.CreditScore = CreditScore
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary
        self.Gender = Gender
        self.Geography = Geography
    
        
        

    def get_data_as_dataframe(self):
        try:
            customer_dict={
            "CreditScore": [self.CreditScore],
            "Age": [self.Age],
            "Tenure": [self.Tenure],
            "Balance": [self.Balance],
            "NumOfProducts": [self.NumOfProducts],
            "HasCrCard": [self.HasCrCard],
            "IsActiveMember": [self.IsActiveMember],
            "EstimatedSalary": [self.EstimatedSalary],
            "Gender": [self.Gender],
            "Geography": [self.Geography]
                 }
            return pd.DataFrame(customer_dict)

        except Exception as e:
            raise Customexception (e,sys)
