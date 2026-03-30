from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the feature columns used during training
with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        form_data = request.form.to_dict()
        
        # Create a dataframe with a single row of zeros matching the model features
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0
        
        # Map user input to our DataFrame
        # In a full app, you'd handle all one-hot encoding exactly as done in training.
        # Here we demonstrate with numeric fields:
        if 'tenure' in form_data and 'tenure' in input_data.columns:
            input_data.at[0, 'tenure'] = float(form_data['tenure'])
        if 'MonthlyCharges' in form_data and 'MonthlyCharges' in input_data.columns:
            input_data.at[0, 'MonthlyCharges'] = float(form_data['MonthlyCharges'])
        if 'TotalCharges' in form_data and 'TotalCharges' in input_data.columns:
            input_data.at[0, 'TotalCharges'] = float(form_data['TotalCharges'])
            
        # Prediction
        prediction = model.predict(input_data)[0]
        
        # Format the result
        result = "Churn (The customer is likely to leave)" if prediction == 1 else "Not Churn (The customer is likely to stay)"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
