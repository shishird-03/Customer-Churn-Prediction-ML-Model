# Customer Churn Prediction Web App

## Short Description
A machine learning project and web application that predicts whether a telecom customer is likely to churn (cancel their service). It features an end-to-end Machine Learning pipeline trained on the IBM Telco dataset and is served through a user-friendly Flask web interface.

---

## 🚀 Features
* **Machine Learning Model**: Built and evaluated multiple models (Logistic Regression, AdaBoost, KNN, Random Forest). The best-performing model (Random Forest) was exported for production.
* **Data Preprocessing**: Handled missing values, outliers (using Z-scores), encoding (One-Hot Encoding), and class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Interactive Web App**: A pure Flask application that allows users to input customer data (like Tenure and Monthly Charges) and instantly receive a prediction on whether the customer will churn.
* **Dynamic UI**: The web frontend automatically calculates total charges based on the inputted tenure and monthly charges to streamline the user experience.

## 🛠️ Technologies Used
* **Python**: Core programming language.
* **Jupyter Notebook**: For Exploratory Data Analysis (EDA) and model building.
* **Scikit-Learn & Imbalanced-Learn**: For training the ML models and handling class imbalances.
* **Flask**: For creating the backend web API and serving the pages.
* **HTML/CSS & JavaScript**: For the frontend user interface.
* **Pandas, NumPy, Seaborn & Matplotlib**: For data manipulation and visualization.

## 📂 Project Structure
* `Model.ipynb`: The core notebook containing data exploration, preprocessing, and model training.
* `app.py`: The Flask server that loads the trained model and handles web requests.
* `templates/index.html`: The frontend HTML interface for the web app.
* `model.pkl`: The exported Random Forest model.
* `model_columns.pkl`: The exported column structure required for the model to make predictions.

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd ML
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install flask pandas scikit-learn imbalanced-learn numpy
   ```

3. **Run the Flask App:**
   ```bash
   python app.py
   ```

4. **Use the App:**
   Open your web browser and go to: `http://127.0.0.1:5000`

## 📊 Dataset Context
The project uses the **Telco Customer Churn** dataset, which contains information about a fictional telecom company that provided home phone and internet services to 7043 customers in California. The target variable is `Churn` (1 = Yes, 0 = No).
