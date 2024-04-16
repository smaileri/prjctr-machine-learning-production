Training procces is in training/model.ipynb

To recreate, just run the notebook.
to have all the libraries run:
`pip install -r requirements.txt`

Stored model is in model/

To run api:
`uvicorn api-server:app --reload`

Then send a request with:
`python3 send-request.py`

I didn't deploy it to aws ec2, because didn't want to use my free-trier or pay for it.

To run container:
cd container-solution
docker build -t api-server-image .
docker run --rm --name api-server -p 8000:80 api-server-build

Then send a POST request with {"text": "SOME_INPUT"} to http://localhost:8080/predict to obtain a prediction.
