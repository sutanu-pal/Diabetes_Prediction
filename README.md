


***End-to-End Diabetes Prediction using Deep Learning***






This project is a complete, end-to-end machine learning application that predicts the onset of diabetes based on clinical parameters. It uses an Artificial Neural Network (ANN) built with TensorFlow/Keras, served via a FastAPI backend, and is fully deployed to the cloud using Render.

## 🚀 Live Demo

*   **Frontend Application:** [https://diabtes-management.onrender.com](https://diabtes-management.onrender.com)
*   **Backend API Docs:** [https://sutanu-diabetes-api.onrender.com/docs](https://sutanu-diabetes-api.onrender.com/docs)

<!-- You can add a screenshot of your application here -->
<!--  -->

## ✨ Features

*   **Deep Learning Model:** Utilizes a Keras Sequential model (ANN) for binary classification.
*   **MLOps Project Structure:** Organized with a clean, modular structure separating training and prediction pipelines.
*   **Custom Logging & Exceptions:** Robust logging for debugging and custom exception handling for clear error reporting.
*   **RESTful API:** A high-performance backend built with FastAPI to serve the model's predictions.
*   **Interactive UI:** A simple, user-friendly frontend built with HTML and Tailwind CSS to interact with the model.
*   **Continuous Deployment:** Fully automated deployment pipeline using GitHub and Render.

## 🛠️ Tech Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **Machine Learning:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
*   **Frontend:** HTML, Tailwind CSS, JavaScript
*   **Deployment:** Git, GitHub, Render

## 📂 Project Structure

The project follows a standard MLOps structure to ensure scalability and maintainability.

```
├── artifacts/
│   ├── model.h5
│   └── preprocessor.joblib
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── app.py
│   ├── logger.py
│   └── exception.py
├── templates/
│   └── index.html
├── requirements.txt
└── setup.py
```

## ⚙️ Local Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sutanu-pal/mlprojects.git
    cd mlprojects
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Train the Model

To train the model from scratch, run the training pipeline. This will generate the trained model (`model.h5`) and the data preprocessor (`preprocessor.joblib`) inside the `artifacts/` directory.

```bash
python src/pipeline/train_pipeline.py
```

### 2. Run the API Server Locally

Once the artifacts are created, you can start the FastAPI server. The `--reload` flag enables hot-reloading for development.

```bash
uvicorn src.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can view the interactive Swagger UI documentation at `http://127.0.0.1:8000/docs`.

### 3. Use the Frontend

Simply open the `index.html` file in your web browser to interact with the prediction UI.

##  FastAPI Endpoint

The application exposes a single API endpoint for making predictions.

*   **URL:** `/predict`
*   **Method:** `POST`
*   **Request Body (JSON):**

    ```json
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
    ```

*   **Success Response (JSON):**

    ```json
    {
      "prediction": "Positive for Diabetes",
      "prediction_value": 1
    }
    ```



*   This project was created by **Sutanu Pal**.
*   Inspiration from various MLOps best practices and tutorials.
