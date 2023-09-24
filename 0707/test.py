import requests as rq

IMAGE_PATH = "./cat.png"

CLASSIFICATION_MODEL_API_URL = "http://localhost:5000/predict"

with open(IMAGE_PATH, 'rb') as f:
    files ={'image':f}
    rqs = rq.post(CLASSIFICATION_MODEL_API_URL, files=files)
    
if rqs.status_code == 200 :
    try : 
        prediction = rqs.json()['predictions']
        print("pred. result >> ", prediction)
    except Exception as e:
        print("API error", str(e))
else :
    print("API error")
    print(rqs.status_code)