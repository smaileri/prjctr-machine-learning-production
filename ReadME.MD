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

AWS Lambda:
cd lambda-func
docker build -t lambda-fastapi .

# Host it in ECR, these are from one of my private

docker tag lambda-fastapi 728300835100.dkr.ecr.us-east-1.amazonaws.com/apilambda:latest
docker push 728300835100.dkr.ecr.us-east-1.amazonaws.com/apilambda:latest

#create lambda with container

Test events:

POST predict
{
"resource": "/mypost",
"path": "/predict",
"httpMethod": "POST",
"requestContext": {},
"multiValueQueryStringParameters": null,
"body": "{\"text\": \"Some data to predict\"}"
}
Health GET /
{
"resource": "/",
"path": "/",
"httpMethod": "GET",
"requestContext": {},
"multiValueQueryStringParameters": null
}
