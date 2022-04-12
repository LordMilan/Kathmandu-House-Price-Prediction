FROM milanmahat/python:2ndtry
COPY . /House-Price-Prediction
WORKDIR /House-Price-Prediction
CMD ["python3","final.py","&"]