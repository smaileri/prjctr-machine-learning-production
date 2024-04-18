from mangum import Mangum
from fastapi import FastAPI, Body
from pydantic import BaseModel
import pickle
import sklearn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Welcome": "Welcome to the FastAPI on Lambda"}


@app.post("/predict")
def retrieve_endpoint(post_data: dict = Body(...)):
    print("predict,post_data ", post_data)
    with open('model/count_vectorizer.pkl', 'rb') as f:
        count_vectorizer = pickle.load(f)

    with open('model/gb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    text = post_data.get('text')
    transformed_text = count_vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    return {"prediction": prediction}


@app.post("/mypost")
def retrieve_endpoint(post_data: dict = Body(...)):
    print("mypost,post_data ", post_data)
    post_data_element1 = post_data["post_data_element1"]
    post_data_element2 = post_data["post_data_element2"]

    print("Post Data Element1: " + post_data_element1)
    print("Post Data Element1: " + post_data_element2)

    return {"response": "Hello Post"}


handler = Mangum(app)
