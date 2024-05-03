import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.classes = ['rice', 'wheat', 'Mung Bean', 'Tea', 'millet', 'maize', 'Lentil', 'Jute',
                        'Coffee', 'Cotton', 'Ground Nut', 'Peas', 'Rubber', 'Sugarcane', 'Tobacco',
                        'Kidney Beans', 'Moth Beans', 'Coconut', 'Black gram', 'Adzuki Beans',
                        'Pigeon Peas', 'Chickpea', 'banana', 'grapes', 'apple', 'mango', 'muskmelon',
                        'orange', 'papaya', 'pomegranate', 'watermelon']
    
    def predict(self,features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            categorical_preds = [self.classes[int(pred)] for pred in preds]
            return categorical_preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    # this is because our html page will get data in the same naming format
    def __init__(  self,
        humidity: float,
        temperature: float,
        ph: float,
        rainfall: float,
        ):

        self.humidity = humidity

        self.temperature = temperature

        self.ph = ph

        self.rainfall = rainfall

    def get_data_as_data_frame(self):    # this is because we have trained model on data frame
        try:
            custom_data_input_dict = {
                "humidity": [self.humidity],
                "temperature": [self.temperature],
                "ph": [self.ph],
                "rainfall": [self.rainfall],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)