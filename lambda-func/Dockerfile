FROM public.ecr.aws/lambda/python:3.10

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN yum update -y
RUN yum update -y python3 curl libcom_err ncurses expat libblkid libuuid libmount
RUN yum install ffmpeg libsm6 libxext6 python3-pip git -y

COPY requirements.txt .
RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY lambda.py ${LAMBDA_TASK_ROOT}/lambda.py
COPY . .

CMD [ "lambda.handler" ]