# 🚀 Health Impact Predictor

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/charolette-xiome/-Health-Impact-Predictor?style=for-the-badge)](https://github.com/charolette-xiome/-Health-Impact-Predictor/stargazers)

[![GitHub forks](https://img.shields.io/github/forks/charolette-xiome/-Health-Impact-Predictor?style=for-the-badge)](https://github.com/charolette-xiome/-Health-Impact-Predictor/network)

[![GitHub issues](https://img.shields.io/github/issues/charolette-xiome/-Health-Impact-Predictor?style=for-the-badge)](https://github.com/charolette-xiome/-Health-Impact-Predictor/issues)

[![GitHub license](https://img.shields.io/github/license/charolette-xiome/-Health-Impact-Predictor?style=for-the-badge)](LICENSE)

**An intuitive web application for predicting potential health impacts based on user-provided inputs using a K-Nearest Neighbors (KNN) machine learning model.**

[Live Demo](https://charolette-xiome.github.io/-Health-Impact-Predictor/index.html) <!-- TODO: Verify or add live demo link -->

</div>

## 📖 Overview

The Health Impact Predictor is a web-based application designed to demonstrate the deployment of a machine learning model for predicting health outcomes. Users can input various health-related parameters through a user-friendly interface, which are then sent to a Python Flask backend. This backend utilizes a pre-trained K-Nearest Neighbors (KNN) model to process the inputs and return a predicted health impact. The result is then displayed on the frontend, offering a clear example of how ML models can be integrated into interactive web applications.

This project serves as a practical showcase for combining frontend web development with a machine learning inference backend.

## ✨ Features

-   🎯 **Predictive Modeling**: Utilizes a pre-trained K-Nearest Neighbors (KNN) model (`knn_mean_model_k1000.pkl`) to infer health impacts.
-   🌐 **Interactive Web Interface**: A clean and responsive HTML/CSS/JavaScript frontend allows users to easily input required data.
-   ⚡ **Real-time Predictions**: Submits user inputs to the backend API and displays predictions instantly without page reloads.
-   ⚙️ **Scalable Backend**: A lightweight Flask API handles prediction requests, making it suitable for integration and further development.
-   📊 **Data Science Integration**: Demonstrates loading and using a `joblib`-serialized machine learning model within a web service.

## 🖥️ Screenshots

<!-- TODO: Add actual screenshots of the application, e.g., input form, prediction results -->
_No screenshots provided. Please add them here._

## 🛠️ Tech Stack

**Frontend:**

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)

![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

**Backend:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)

**Machine Learning:**

![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

![Joblib](https://img.shields.io/badge/Joblib-FF6F00?style=for-the-badge&logo=python&logoColor=white)

## 🚀 Quick Start

Follow these steps to get the Health Impact Predictor up and running on your local machine.

### Prerequisites
-   **Python 3.x**: Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
-   **pip**: Python's package installer, usually comes with Python.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/charolette-xiome/-Health-Impact-Predictor.git
    cd -Health-Impact-Predictor
    ```

2.  **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    This will install Flask, scikit-learn, numpy, pandas, and joblib.

3.  **Ensure model file is present**
    The pre-trained model `knn_mean_model_k1000.pkl` should be in the root directory. It is included in the repository.

### Start development server

1.  **Run the Flask backend**
    ```bash
    python app.py
    ```
    The server will typically run on `http://127.0.0.1:5000/`.

2.  **Open your browser**
    Navigate to `http://localhost:5000/` (or `http://127.0.0.1:5000/`) in your web browser. The `index.html` file will be served by the Flask application, and `script.js` will handle interactions with the backend.

## 📁 Project Structure

```
.
├── README.md                 # This README file
├── app.py                    # Flask backend application, handles ML model loading and prediction API
├── index.html                # Main frontend page with user input form
├── script.js                 # Frontend JavaScript for handling user input, API calls, and displaying results
├── style.css                 # Frontend styling for the web interface
├── knn_mean_model_k1000.pkl  # Pre-trained K-Nearest Neighbors (KNN) machine learning model
├── requirements.txt          # Python dependencies
└── website.html              # (Optional) Another HTML file, potentially an alternative or secondary page
```

## 📚 API Reference

The Flask backend provides a single API endpoint for model inference.

### `/predict`

-   **URL**: `/predict`
-   **Method**: `POST`
-   **Description**: Accepts a JSON payload of health-related features, processes them using the loaded KNN model, and returns a predicted health impact.

**Request Body Example:**
The exact structure of the input features depends on the training data of the `knn_mean_model_k1000.pkl`. Based on common practices, `script.js` likely collects values from a form and sends them as a dictionary.

```json
{
    "feature1": 10,
    "feature2": 25.5,
    "feature3": 0,
    "feature4": 1,
    "...": "..."
}
```
*(Note: Replace `feature1`, `feature2`, etc., with the actual feature names and expected data types used by your model.)*

**Response Body Example (Success):**
```json
{
    "prediction": 5
}
```
*(Note: The `prediction` value will be an integer representing the inferred health impact.)*

**Response Body Example (Error):**
```json
{
    "error": "Input data format is incorrect or prediction failed."
}
```

## 🔧 Development

### Available Scripts
-   **`python app.py`**: Starts the Flask development server, hosting both the backend API and the static frontend files (`index.html`, `script.js`, `style.css`).

### Development Workflow
1.  Modify frontend files (`index.html`, `script.js`, `style.css`) and refresh your browser to see changes.
2.  Modify backend files (`app.py`) and restart the `python app.py` command for changes to take effect.
3.  If `requirements.txt` changes, reinstall dependencies using `pip install -r requirements.txt`.

## 🤝 Contributing

We welcome contributions to enhance the Health Impact Predictor! If you have suggestions or improvements, please consider opening an issue or submitting a pull request.

### Development Setup for Contributors
The development setup is the same as the quick start guide. Ensure you have Python and `pip` installed, then clone the repository and install dependencies.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. <!-- TODO: Verify license or create a LICENSE file -->

## 🙏 Acknowledgments

-   **Flask**: For providing a lightweight web framework for Python.
-   **Scikit-learn**: For robust machine learning algorithms and tools.
-   **NumPy & Pandas**: Essential libraries for numerical operations and data manipulation.
-   **Joblib**: For efficiently saving and loading Python objects, especially large NumPy arrays.

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/charolette-xiome/-Health-Impact-Predictor/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [charolette-xiome](https://github.com/charolette-xiome)

</div>

