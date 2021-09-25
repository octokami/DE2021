FROM python:3.8.5-slim-buster

LABEL maintainer="Peng Xiao<xiaoquwl@gmail.com>"

COPY . /

RUN pip install -r requirements.txt && pip install gunicorn

RUN ["chmod", "+x", "/runserver.sh"]

ENTRYPOINT ["/runserver.sh"]