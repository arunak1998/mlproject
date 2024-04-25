import os
import sys
from src.logger import logging
from src.exception import  Customexception

import pandas as pd

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformerConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass

@dataclass
class DataInjestionConfig:
    raw_data_path:str=os.path.join('artifacts','data.csv')
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')


class DataInjestion:
    def __init__(self):
        self.injestion_config=DataInjestionConfig()

    def initiate_data_injestion(self):
        logging.info("ENtered the Data Injextion Method or Componet")

        try:
            df=pd.read_csv('notebook\Churn_modelling.csv')
            logging.info('Read the Dtaset as DataFrame')
            os.makedirs(os.path.dirname(self.injestion_config.test_data_path),exist_ok=True)
            df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)

            logging .info('Test Train data split Initiated')
            train_data,test_data=train_test_split(df,test_size=0.2,random_state=42)

            train_data.to_csv(self.injestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.injestion_config.test_data_path,index=False,header=True)


            logging.info('Injestion of Dta is completed')

            return(
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )



        except Exception as e:
            raise Customexception(e,sys)


if __name__=='__main__':
    obj=DataInjestion()
    train_data,test_data=obj.initiate_data_injestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_train=ModelTrainer()
    print(model_train.initiate_model_trainer(train_arr,test_arr))