*****End-to-End Diabetes Prediction using Deep Learning*****



This project is a complete, end-to-end machine learning application that predicts the onset of diabetes based on clinical parameters. It uses an Artificial Neural Network (ANN) built with TensorFlow/Keras, served via a FastAPI backend, and is fully deployed to the cloud using Render.

🚀 Live Demo
Frontend Application: https://diabtes-management.onrender.com

Backend API Docs: https://sutanu-diabetes-api.onrender.com/docs

✨ Features
Deep Learning Model: Utilizes a Keras Sequential model (ANN) for binary classification.

MLOps Project Structure: Organized with a clean, modular structure separating training and prediction pipelines.

Custom Logging & Exceptions: Robust logging for debugging and custom exception handling for clear error reporting.

RESTful API: A high-performance backend built with FastAPI to serve the model's predictions.

Interactive UI: A simple, user-friendly frontend built with HTML and Tailwind CSS to interact with the model.

Continuous Deployment: Fully automated deployment pipeline using GitHub and Render.

🛠️ Tech Stack
Backend: Python, FastAPI, Uvicorn

Machine Learning: TensorFlow, Keras, Scikit-learn, Pandas, NumPy

Frontend: HTML, Tailwind CSS, JavaScript

Deployment: Git, GitHub, Render

📂 Project Structure
The project follows a standard MLOps structure to ensure scalability and maintainability.


├── artifacts/              # Stores trained model (model.h5) and preprocessor

├── src/

│   ├── pipeline/

│   │   ├── train_pipeline.py   # Script to run the full training process

│   │   └── predict_pipeline.py # Classes for loading model and making predictions

│   ├── app.py                  # The FastAPI application server

│   ├── logger.py               # Custom logging setup

│   └── exception.py            # Custom exception handling

├── index.html              # The frontend user interface

├── requirements.txt        # Project dependencies

└── setup.py                # For setting up the project as a local package


⚙️ Setup and Installation
To run this project locally, follow these steps:

1. Clone the repository:

git clone https://github.com/sutanu-pal/mlprojects.git
cd mlprojects


2. Create and activate a virtual environment:

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate


3. Install the required dependencies:

pip install -r requirements.txt


🚀 Usage
1. Train the Model
To train the model from scratch, run the training pipeline. This will create the artifacts folder with the model.h5 and preprocessor.joblib files.

python src/pipeline/train_pipeline.py


2. Run the API Server Locally
Once the artifacts are created, you can start the FastAPI server.

uvicorn src.app:app --reload


The API will be available at http://127.0.0.1:8000. You can view the interactive documentation at http://127.0.0.1:8000/docs.

3. Use the Frontend
Simply open the index.html file in your web browser to use the user interface.

FastAPI Endpoint
The application exposes a single API endpoint for predictions.

URL: /predict

Method: POST

Request Body (JSON):

{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}


Success Response (JSON):

{
  "prediction": "Positive for Diabetes",
  "prediction_value": 1
}

