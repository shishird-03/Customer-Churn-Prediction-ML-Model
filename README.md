# Customer Churn Prediction & Web Dashboard

This project is an end-to-end Machine Learning solution to predict customer churn. It includes an exploratory data analysis and model training pipeline, as well as an interactive web dashboard built with Flask to serve the predictions and provide insights through SHAP explainability and Plotly visualizations.

##  Features

- **Machine Learning Model**: A robust Random Forest Classifier trained to balance the classes using SMOTE.
- **Explainable AI (XAI)**: Integration with SHAP to explain the key features influencing each prediction.
- **Web Dashboard**: An interactive, user-friendly UI built with Flask to input customer data and get real-time churn predictions.
- **REST API**: JSON-based endpoints for headless predictions (`/predict-api`).
- **Comprehensive Evaluation**: Extensive model validation handling class imbalances, achieving high prediction accuracy.
- **Batch Processing:** Ability to upload CSVs and get mass predictions.

##  Tech Stack

- **Data Processing & ML**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`
- **Model Explainability**: `shap`
- **Web Framework**: `Flask`, `werkzeug`, `gunicorn`
- **Visualizations**: `plotly`
- **Model Serialization**: `pickle`

##  Project Structure

```text
.
├── app_enhanced.py                 # Flask web application and API endpoints
├── Model.ipynb                     # Jupyter notebook for EDA, data preprocessing, and model training
├── customer_churn_prediction.csv   # Primary dataset
├── evaluation_report.md            # Detailed model evaluation and metrics
├── project_presentation_guide.md   # Guide for presenting the project
├── requirements.txt                # Python dependencies
└── templates/                      # HTML templates for the Flask application
    ├── dashboard.html
    ├── index_enhanced.html
    └── model_info.html
```

##  Installation & Setup

1. **Clone the repository** (if applicable) and navigate to the project directory:
   ```bash
   cd ML
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On macOS/Linux: source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**:
   Run the `Model.ipynb` notebook to preprocess the data, train the Random Forest model, and export the required `.pkl` files (`model.pkl` and `model_columns.pkl`) needed by the web app.

5. **Run the Flask Application**:
   ```bash
   python app_enhanced.py
   ```
   The application will start at `http://localhost:5000` or `http://127.0.0.1:5000`.

##  Model Performance

Based on the evaluation report, the Random Forest model performs exceptionally well at identifying churn patterns:

- **Overall Accuracy**: ~94.39%
- **Precision (Churn)**: 0.90
- **Recall (Churn)**: 0.89
- **Class Balancing**: Used SMOTE to effectively manage false positives and false negatives, ensuring active users are properly prioritized for retention campaigns.

*See `evaluation_report.md` for a complete breakdown of metrics and confusion matrix analysis.*

##  API Usage

You can make programmatic requests to the API for predictions:

**Endpoint:** `POST /predict-api`

**Example Payload:**
```json
{
  "tenure": 12,
  "MonthlyCharges": 75.50,
  "TotalCharges": 906.00,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "OnlineSecurity": "No"
}
```
Author: Shishir D
