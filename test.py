import requests
import json
import pandas as pd

url = 'http://localhost:5000/test/'

data = 'https://bc-clinic.ru/photogallery/dermatology/vospalilas_rodinka.jpg'

j_data = {'name':data}
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.get(url = url, params = j_data)
# ans = r.json()
print(r.text)