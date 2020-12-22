import requests

req=requests.post("https://pytorch-flask-app.herokuapp.com/predict",files={"file":open("./seven.png",'rb')})

print(req.text)