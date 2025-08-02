from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os
app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open('ridge.pkl','rb'))
standard_scaler = pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/home")
def home_page():
    return render_template("home.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            # Collect form data
            posted_by = int(request.form['posted_by'])
            rera = int(request.form['rera'])
            bhk = int(request.form['bhk'])
            lat = float(request.form['lat'])
            lon = float(request.form['lon'])
            
            # Create input array with the correct features used in the model
            # Features used: ['Unnamed: 0.1', 'RERA', 'BHK_NO.', 'LONGITUDE', 'LATITUDE', 'transformed_price']
            input_data = np.array([[0, rera, bhk, lon, lat, 0]])  # Unnamed: 0.1 = 0, transformed_price = 0
            
            # Scale the input data
            input_scaled = standard_scaler.transform(input_data)
            
            # Make prediction
            prediction = ridge_model.predict(input_scaled)[0]
            
            # Convert prediction to price in lacs
            predicted_price = round(prediction, 2)
            
            return render_template("prediction.html", 
                                predicted_price=predicted_price,
                                posted_by=posted_by,
                                rera=rera,
                                bhk=bhk,
                                lat=lat,
                                lon=lon)

        except Exception as e:
            return f"Something went wrong: {str(e)}"
    
    # GET request - show the form
    return render_template("prediction.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
