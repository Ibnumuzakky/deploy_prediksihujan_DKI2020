import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, template_folder = 'template' )

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    features = [float(i) for i in request.form.values()]
    array_features = [np.array(features)]
    prediction = model.predict(array_features)
    
    return render_template('index.html', result = str(*prediction))

if __name__ == '__main__':
    app.run(debug=True)
