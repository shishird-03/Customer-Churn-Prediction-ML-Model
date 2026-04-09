from flask import Flask, request, render_template, jsonify, send_file
import pickle
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import shap
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ==================== LOAD MODEL & DATA ====================
with open('model.pkl', 'rb') as f:
    rfc_best = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Load training data for SHAP explanations (sample for performance)
try:
    df_train = pd.read_csv('customer_churn_prediction.csv')
    # Note: In production, you'd load the processed dataset used in training
except:
    df_train = None

# ==================== FEATURE CATEGORIES ====================
CATEGORICAL_FEATURES = {
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'OnlineSecurity': ['Yes', 'No'],
    'TechSupport': ['Yes', 'No'],
    'OnlineBackup': ['Yes', 'No'],
    'DeviceProtection': ['Yes', 'No'],
    'StreamingTV': ['Yes', 'No'],
    'StreamingMovies': ['Yes', 'No'],
}

NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# ==================== PREDICTION HISTORY ====================
prediction_history = []

# ==================== HELPER FUNCTIONS ====================
def encode_features(form_data):
    """Convert form data to properly encoded dataframe matching model training"""
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0
    
    # Handle numeric features
    for feature in NUMERIC_FEATURES:
        if feature in form_data and feature in input_data.columns:
            try:
                input_data.at[0, feature] = float(form_data[feature])
            except:
                pass
    
    # Handle categorical features (one-hot encoding)
    for feature, options in CATEGORICAL_FEATURES.items():
        value = form_data.get(feature, None)
        if value:
            # Convert to one-hot encoded columns
            for option in options[:-1]:  # Skip last (drop_first=True in training)
                col_name = f"{feature}_{option.replace(' ', '_')}"
                if col_name in input_data.columns:
                    input_data.at[0, col_name] = 1 if value == option else 0
    
    return input_data

def get_prediction_details(input_data):
    """Get prediction with probability and confidence"""
    prediction = rfc_best.predict(input_data)[0]
    probability = rfc_best.predict_proba(input_data)[0]
    
    confidence = max(probability) * 100
    churn_probability = probability[1] * 100  # Probability of churn
    
    return {
        'prediction': int(prediction),
        'confidence': round(confidence, 2),
        'churn_probability': round(churn_probability, 2),
        'no_churn_probability': round(probability[0] * 100, 2)
    }

def generate_shap_explanation(input_data):
    """Generate SHAP values for feature importance explanation"""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rfc_best)
        shap_values = explainer.shap_values(input_data)[1]  # Get SHAP values for churn class
        
        # Get top 10 features affecting prediction
        importance_df = pd.DataFrame({
            'Feature': input_data.columns,
            'SHAP_Value': np.abs(shap_values),
            'Direction': ['↑ Increases Churn' if val > 0 else '↓ Reduces Churn' 
                         for val in shap_values]
        }).sort_values('SHAP_Value', ascending=False).head(10)
        
        return importance_df.to_dict('records')
    except Exception as e:
        return []

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Main prediction page"""
    return render_template('index_enhanced.html', 
                         categorical_features=CATEGORICAL_FEATURES,
                         numeric_features=NUMERIC_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        form_data = request.form.to_dict()
        
        # Encode features properly
        input_data = encode_features(form_data)
        
        # Get prediction with confidence
        details = get_prediction_details(input_data)
        
        # Generate SHAP explanations
        shap_explanation = generate_shap_explanation(input_data)
        
        # Store in history
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input': form_data,
            'prediction': details['prediction'],
            'confidence': details['confidence'],
            'churn_probability': details['churn_probability']
        }
        prediction_history.append(prediction_record)
        
        return jsonify({
            'success': True,
            'prediction': details['prediction'],
            'confidence': details['confidence'],
            'churn_probability': details['churn_probability'],
            'no_churn_probability': details['no_churn_probability'],
            'message': 'High Risk - Customer likely to churn' if details['prediction'] == 1 
                      else 'Low Risk - Customer likely to stay',
            'recommendation': 'Proactive retention offer recommended' if details['prediction'] == 1 
                            else 'Maintain current service quality',
            'shap_explanation': shap_explanation
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict-api', methods=['POST'])
def predict_api():
    """API endpoint with JSON input"""
    try:
        data = request.get_json()
        
        # Create dataframe from JSON input
        input_df = pd.DataFrame([data])
        
        # Ensure columns match model
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[model_columns]
        
        details = get_prediction_details(input_df)
        
        return jsonify({
            'success': True,
            'prediction': details['prediction'],
            'confidence': details['confidence'],
            'churn_probability': details['churn_probability']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction from CSV file"""
    try:
        print(f"FILES: {request.files}, FORM: {request.form}")
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'File must be CSV'}), 400
        
        # Read CSV
        stream = StringIO(file.stream.read().decode("UTF8"), newline=None)
        df = pd.read_csv(stream)
        
        # Prepare data
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        
        df_prepared = df[model_columns]
        
        # Make predictions
        predictions = rfc_best.predict(df_prepared)
        probabilities = rfc_best.predict_proba(df_prepared)
        
        # Create results dataframe
        results_df = df.copy() if len(df.columns) <= 20 else pd.DataFrame()
        results_df['Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
        results_df['Churn_Probability_%'] = (probabilities[:, 1] * 100).round(2)
        results_df['Confidence_%'] = (np.max(probabilities, axis=1) * 100).round(2)
        
        # Convert to CSV for download
        output = BytesIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard"""
    try:
        if len(prediction_history) == 0:
            return render_template('dashboard.html', 
                                 total_predictions=0,
                                 churn_rate=0,
                                 avg_confidence=0,
                                 charts_data=None)
        
        # Analytics
        df_history = pd.DataFrame(prediction_history)
        total_preds = len(df_history)
        churn_count = (df_history['prediction'] == 1).sum()
        churn_rate = (churn_count / total_preds * 100) if total_preds > 0 else 0
        avg_confidence = df_history['confidence'].mean()
        
        # Churn distribution chart
        churn_labels = ['Churn', 'No Churn']
        churn_values = [churn_count, total_preds - churn_count]
        
        fig_pie = go.Figure(data=[go.Pie(labels=churn_labels, values=churn_values,
                                         marker=dict(colors=['#ff6b6b', '#51cf66']))])
        fig_pie.update_layout(title="Churn Distribution", height=400)
        
        # Confidence distribution
        fig_hist = go.Figure(data=[go.Histogram(x=df_history['confidence'], nbinsx=20,
                                               marker=dict(color='#4c72b0'))])
        fig_hist.update_layout(title="Model Confidence Distribution", 
                              xaxis_title="Confidence %", yaxis_title="Count", height=400)
        
        # Trend over time
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        df_history_sorted = df_history.sort_values('timestamp')
        df_history_sorted['cumulative_churn'] = (df_history_sorted['prediction'] == 1).cumsum()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=df_history_sorted['timestamp'], 
                                      y=df_history_sorted['cumulative_churn'],
                                      mode='lines+markers', name='Cumulative Churns'))
        fig_trend.update_layout(title="Churn Predictions Trend", 
                               xaxis_title="Time", yaxis_title="Cumulative Churns", height=400)
        
        return render_template('dashboard.html',
                             total_predictions=total_preds,
                             churn_rate=round(churn_rate, 2),
                             avg_confidence=round(avg_confidence, 2),
                             churn_count=churn_count,
                             pie_chart=fig_pie.to_html(include_plotlyjs=False, div_id="pie-chart"),
                             histogram=fig_hist.to_html(include_plotlyjs=False, div_id="histogram"),
                             trend=fig_trend.to_html(include_plotlyjs=False, div_id="trend"))
    
    except Exception as e:
        return render_template('dashboard.html', error=str(e))

@app.route('/model-info')
def model_info():
    """Model performance and feature information"""
    feature_importance = pd.DataFrame({
        'Feature': model_columns,
        'Importance': rfc_best.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    return render_template('model_info.html',
                         total_features=len(model_columns),
                         top_features=feature_importance.to_dict('records'),
                         model_type='Random Forest (Tuned)',
                         test_accuracy='90.24%',
                         precision='87.2%',
                         recall='91.8%')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': rfc_best is not None})

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    # Check if all required files exist
    if not os.path.exists('model.pkl') or not os.path.exists('model_columns.pkl'):
        print("ERROR: Model files not found. Run Model.ipynb first!")
    else:
        print("✓ Model loaded successfully")
        print("✓ Starting Flask app with enhanced features...")
        print("\nAvailable endpoints:")
        print("  - http://localhost:5000/ (Main UI)")
        print("  - http://localhost:5000/dashboard (Analytics Dashboard)")
        print("  - http://localhost:5000/model-info (Model Information)")
        print("  - POST /predict (Single prediction)")
        print("  - POST /batch-predict (CSV batch predictions)")
        print("  - POST /predict-api (JSON API)")
        app.run(debug=True, port=5000)
