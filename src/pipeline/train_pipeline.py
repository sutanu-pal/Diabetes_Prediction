import os
import sys
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.exception import CustomException
from src.logger import logging

# Configuration class for file paths
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.h5")
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.joblib")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self):
        logging.info("Starting model training process")
        try:
            # --- 1. Data Ingestion and Preparation ---
            logging.info("Loading dataset")
            # Load the dataset from the URL
            url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            df = pd.read_csv(url, header=None, names=column_names)
            logging.info("Dataset loaded successfully")

            # Separate features (X) and target (y)
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            logging.info("Data split into training and test sets")

            # --- 2. Data Transformation (Preprocessing) ---
            logging.info("Initializing StandardScaler for preprocessing")
            preprocessor = StandardScaler()
            
            # Fit the scaler on the training data and transform it
            X_train_scaled = preprocessor.fit_transform(X_train)
            
            # Transform the test data using the same fitted scaler
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Data scaling complete")

            # --- 3. Model Building and Training ---
            logging.info("Building the Keras Sequential model")
            model = Sequential([
                Dense(12, input_dim=X_train_scaled.shape[1], activation='relu'),
                Dense(8, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            logging.info("Model compiled successfully")
            
            # Use EarlyStopping to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            logging.info("Starting model training")
            history = model.fit(
                X_train_scaled, 
                y_train, 
                epochs=100, 
                batch_size=10, 
                validation_split=0.2, 
                callbacks=[early_stopping],
                verbose=1
            )
            logging.info("Model training complete")

            # --- 4. Saving Artifacts ---
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            
            # Save the trained model
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"Trained model saved to {self.model_trainer_config.trained_model_file_path}")

            # Save the preprocessor object
            joblib.dump(preprocessor, self.model_trainer_config.preprocessor_obj_file_path)
            logging.info(f"Preprocessor object saved to {self.model_trainer_config.preprocessor_obj_file_path}")

            # --- 5. Evaluation (Optional but good practice) ---
            logging.info("Evaluating model on test data")
            loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
            logging.info(f"Test Accuracy: {accuracy:.4f}")
            
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)

# This block allows you to run the training pipeline directly
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_training()
