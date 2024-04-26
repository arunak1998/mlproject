import sys
import os
from dataclasses import dataclass
from src.utils import save_obj
from src.utils import OutlierRemover,CustomOneHotEncoder
import numpy as np
from sklearn.compose import make_column_selector
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  OneHotEncoder
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import Customexception
from sklearn.preprocessing import FunctionTransformer
import sklearn



@dataclass
class DataTransformerConfig:
    preprocessor_obj_file=os.path.join("artifacts","preprocessor.pkl")
    


class DataTransformation:
    def __init__(self) :
        self.data_transformation_config=DataTransformerConfig()

    def get_data_transformer_object(self):
        '''
        This Function is Responsible for Data Transformation
        '''
        try:
            categorical_features=['Gender','Geography']
             
            numerical_feature=[
                'CreditScore',
                'Age',               
                'Tenure',            
                'Balance',          
                'NumOfProducts',    
                'HasCrCard' ,       
                'IsActiveMember',   
                'EstimatedSalary']
            
            
           
            logging.info('Categorical_features Encoding is Completed')
           
            


            preprocessor = Pipeline([
             ('custom_encoder', CustomOneHotEncoder(categorical_columns=categorical_features)),
    
                     ])
           
           
            return preprocessor
    
        except Exception as e:
            raise Customexception(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            train_data.drop(columns=['CustomerId','Surname','RowNumber'],inplace=True,axis=1)
            test_data.drop(columns=['CustomerId','Surname','RowNumber'],inplace=True,axis=1)

            logging.info('Read Train and test data Completed')

            logging.info('Obtaining PrepProcesing Object')

            

            preprocessing_object=self.get_data_transformer_object()

            target_column="Exited"
            numerical_feature=[
                'CreditScore',
                'Age',               
                'Tenure',            
                'Balance',          
                'NumOfProducts',    
                'HasCrCard' ,       
                'IsActiveMember',   
                'EstimatedSalary']
            
            for column_name in numerical_feature:
    # Apply outlier removal process to the data for the current column
             
              train_data = OutlierRemover(col=column_name).fit_transform(train_data)

            for column_name in numerical_feature:
    # Apply outlier removal process to the data for the current column
              test_data = OutlierRemover(col=column_name).fit_transform(test_data)
            logging.info("f Outliers has been removed from Training and Test data ")

            

           
           
            input_training_feature=train_data.drop(columns=['Exited'],axis=1)
           
            target_feature_train_df=train_data[target_column]

            input_test_feature=test_data.drop(columns=['Exited'],axis=1)
            target_feature_test_df=test_data[target_column]
            
            logging.info("f Preprocessing has been applied  Training and Test data ")

            print(input_test_feature.shape)
            # encoder = CustomOneHotEncoder(categorical_columns=['Geography', 'Gender'])
            # input_feature_train_arr= encoder.fit_transform(input_training_feature)
            
            # encoder = CustomOneHotEncoder(categorical_columns=['Geography', 'Gender'])
            # input_feature_test_arr= encoder.fit_transform(input_test_feature)
            
            input_feature_train_arr=preprocessing_object.fit_transform(input_training_feature)
            input_feature_test_arr= preprocessing_object.transform(input_test_feature)
            #
            logging.info("f One Hot ENcoding has been applied  Training and Test data ")

          
            
            

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            print(test_arr.shape)
            logging.info('Preprocessing Completed')
           
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_object
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
        

        except Exception as e:
            raise Customexception(e,sys)
        

    



    
