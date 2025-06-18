# Banknote Authenticity Detection API

This project is a FastAPI-based web application that uses a machine learning model to determine whether a banknote is genuine or forged. It provides a RESTful API endpoint for real-time classification based on statistical features extracted from scanned banknote images.

## Overview

The model is trained on digitized features such as variance, skewness, kurtosis, and entropy of the wavelet-transformed images of banknotes. The trained logistic regression model is then served using FastAPI, allowing external systems or users to make predictions via simple POST requests.

## Features

- Machine learning model trained using scikit-learn
- Fast and lightweight API with FastAPI
- Predicts the authenticity of banknotes using four key statistical features
- JSON-based input and output
- Model saved using Pickle

## Technologies Used

- Python 3
- scikit-learn
- FastAPI
- Uvicorn
- Pickle

## API Endpoint

### `POST /predict`

Send a JSON payload containing the following fields:

#### Input JSON

```json
{
  "variance": 2.3,
  "skewness": 1.5,
  "kurtosis": 0.1,
  "entropy": -1.2
}

## RUN the API
uvicorn app:app --reload

"# Banknote-Authenticity-Detection-API" 
"# Banknote-Authenticity-Detection-API" 
