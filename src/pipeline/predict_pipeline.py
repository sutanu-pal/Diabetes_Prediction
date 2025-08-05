import os
import sys
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        # Define paths to the trained model and preprocessor
        self.model_path = os.path.join("artifacts", "model.h5")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.joblib")
        logging.info("Prediction pipeline initialized.")

    def predict(self, features):
        """
        Loads the model and preprocessor to make predictions on new data.
        :param features: A pandas DataFrame with the same columns as the training data.
        :return: A numpy array of predictions.
        """
        try:
            logging.info("Loading model and preprocessor for prediction")
            # Load the saved model and preprocessor objects
            model = load_model(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)
            logging.info("Artifacts loaded successfully")

            # Scale the input features using the loaded preprocessor
            scaled_features = preprocessor.transform(features)
            logging.info("Input data scaled successfully")

            # Make predictions
            predictions_prob = model.predict(scaled_features)
            
            # Convert probabilities to binary output (0 or 1)
            predictions = (predictions_prob > 0.5).astype(int)
            logging.info("Prediction complete")
            
            return predictions

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    """
    This class is responsible for mapping the input data from an HTML form
    or API request to a pandas DataFrame.
    """
    def __init__(self,
                 pregnancies: int,
                 glucose: int,
                 blood_pressure: int,
                 skin_thickness: int,
                 insulin: int,
                 bmi: float,
                 diabetes_pedigree_function: float,
                 age: int):
        
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.blood_pressure = blood_pressure
        self.skin_thickness = skin_thickness
        self.insulin = insulin
        self.bmi = bmi
        self.diabetes_pedigree_function = diabetes_pedigree_function
        self.age = age

    def get_data_as_dataframe(self):
        """
        Returns the input data as a pandas DataFrame.
        """
        try:
            custom_data_input_dict = {
                "Pregnancies": [self.pregnancies],
                "Glucose": [self.glucose],
                "BloodPressure": [self.blood_pressure],
                "SkinThickness": [self.skin_thickness],
                "Insulin": [self.insulin],
                "BMI": [self.bmi],
                "DiabetesPedigreeFunction": [self.diabetes_pedigree_function],
                "Age": [self.age],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)