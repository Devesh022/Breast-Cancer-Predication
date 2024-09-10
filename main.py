from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = [float(x) for x in request.form.values()]
    final_features = np.array([data])
    prediction = model.predict(final_features)
    output = 'Malignant' if prediction[0] == 1 else 'Benign'
    return render_template('index.html', prediction_text=f'The tumor is {output}')

if __name__ == "__main__":
    app.run(debug=True)
