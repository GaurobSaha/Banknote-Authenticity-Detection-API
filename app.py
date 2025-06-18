# Required Libraries
# .venv\Scripts\activate.bat

import uvicorn ##ASGI
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

# Creating app object
app=FastAPI()
pickle_in = open("classifier.pkl", "rb") #opens the file named classifier.pkl in read-binary ("rb") mode
classifier = pickle.load(pickle_in) #deserializes contents of .pkl file and loads the actual model into memory as a Python object 

# Index route, opens the home page
@app.get('/')
def index():
    return {'message': 'Welcome to the Bank Note Authentication API'}

# Route with a single parameter,returns the parameter with a message
# loacted at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello {name}, welcome to the Bank Note Authentication API'}

# Expose the prediction functionality, make a prediction from the passed data
# JSON data and returns the predicted Bank Note with the confidence score

@app.post('/predict')
def predict_bank_note(data: BankNote): #type hinting, a Python feature that helps clarify what kind of data a parameter should be.
    #This function takes a parameter called data, and it must be an instance of the BankNote class.
    data = data.model_dump()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    
    # Make prediction
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    
    if prediction[0]<0.5:
        prediction="The Note is Authentic"
    else:
        prediction="The Note is Fake"

    return {'prediction': prediction}

#Run the API using uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000) 

# uvicorn app:app --reload
# uvicorn the file name: the app object name --reload
# The --reload flag enables auto-reloading of the server when code changes are detected