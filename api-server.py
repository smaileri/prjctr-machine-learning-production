from fastapi import FastAPI
import pickle
from fastapi import Request


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingRegressor
import json
with open('model/count_vectorizer.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

with open('model/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)


app = FastAPI()
@app.post("/predict")
async def predict(request: Request): 
    body = await request.body()
    data = json.loads(body)
    text = data.get('text')
    transformed_text = count_vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return {"prediction": prediction} 
