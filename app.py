from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

app = FastAPI()
model = joblib.load("model.pkl")

train = pd.read_csv("training.csv")
X_train = train.drop(columns=["ID", "mobile_money_classification",
                               "mobile_money", "savings", "borrowing", "insurance"])
y_train = train["mobile_money_classification"]

y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)

@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <h1>Tanzania Financial Inclusion Predictor</h1>
    <form action="/predict" method="post">
        <p>Age: <input name="Q1" type="number"></p>
        <p>Gender (1=Male, 2=Female): <input name="Q2" type="number"></p>
        <p>Owns Phone (1=Yes, 2=No): <input name="Q7" type="number"></p>
        <p>Education: <input name="Q4" type="number"></p>
        <small>
            1=No formal education,
            2=Some primary,
            3=Primary completed,
            4=Technical training,
            5=Some secondary,
            6=University
        </small>
        <br><br>
        <button type="submit">Predict</button>
    </form>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(Q1: int = Form(...), Q2: int = Form(...),
                  Q7: int = Form(...), Q4: int = Form(...)):
    features = [-1] * 31
    features[0] = Q1
    features[1] = Q2
    features[3] = Q4
    features[6] = Q7

    proba = model.predict_proba([features])[0]
    classes = ["No services", "Other only", "MM only", "MM plus"]
    result = classes[np.argmax(proba)]
    confidence = np.max(proba)

    return f"""
    <h1>Result: {result}</h1>
    <p>Confidence: {confidence:.2%}</p>
    <p>No services: {proba[0]:.2%}</p>
    <p>Other only: {proba[1]:.2%}</p>
    <p>MM only: {proba[2]:.2%}</p>
    <p>MM plus: {proba[3]:.2%}</p>
    <hr>
    <h3>Model Accuracy: {accuracy:.2%}</h3>
    <a href="/">Go back</a>
    """