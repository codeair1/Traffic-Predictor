# %%
import joblib 
import pandas as pd
from datetime import datetime 
import os
import requests
import joblib



def TrafficPred(lag1,lag2):

    FILE_ID = '12r1E3v1oaTdBlz51VVlEZ-ZqSP2Vn8jX'
    MODEL_URL = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    LOCAL_PATH = '/tmp/traffic_model.pkl'

    def download_model():
        if not os.path.exists(LOCAL_PATH):
            response = requests.get(MODEL_URL)
            with open(LOCAL_PATH, "wb") as f:
                f.write(response.content)
        return joblib.load(LOCAL_PATH)

    model = download_model()

    current_datetime = datetime.now()

    hour = current_datetime.hour + 1


    weekday_map = {
            'Monday': 0,
            'Tuesday': 1,
            'Wednesday': 2,
            'Thursday': 3,
            'Friday': 4,
            'Saturday': 5,
            'Sunday': 6
        }
    day_of_week1 = current_datetime.strftime("%A")
    day_of_week = weekday_map[day_of_week1]

    data = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'lag1': lag1,
            'lag2': lag2
        }])


    prediction = model.predict(data)
    return prediction[0]


