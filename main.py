import pycountry
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

from api_request import GoogleRequests
from filter_data import correlation_filter
from models import gradient_descent, predict
from process_data import convert_to_weekly, get_mean_from_csv, plot_result, preprocess_data

plt.style.use('ggplot')

def get_anchortime():
    #"2021-01-01 2022-01-01"
    today_date = date.today()
    past_date = today_date - timedelta(days=365)
    anchor_time = str(past_date) + " " + str(today_date)
    #return anchor_time
    return "2020-01-20 2022-02-28"

COUNTRY = "united states"
KEYWORDS = ['stomach pain']
CAT = '0'
TIMEFRAMES = ['today 12-m', 'today 3-m', 'today 1-m']
GPROP = ''
ANCHOR_TIME = get_anchortime()
PATH_TO_CSV = 'covid_deaths_usafacts.csv'

VALUE = 'covid_mean'
google_requests = GoogleRequests(KEYWORDS, CAT, TIMEFRAMES, COUNTRY, GPROP, ANCHOR_TIME)

data = google_requests.request_window()
data = correlation_filter(data, KEYWORDS)

covid_data = get_mean_from_csv(PATH_TO_CSV)
weekly_covid_data = convert_to_weekly(covid_data)

frames = [data, weekly_covid_data]
result_array = pd.concat(frames, axis=1)
result_array = correlation_filter(result_array, [VALUE])

X, y = preprocess_data(result_array, VALUE)
w, b, cost_list, epoch_list = gradient_descent(x=X, y=y, w=np.zeros(X.shape[1]), b=0, learning_rate=0.001, epochs=50000)
y_pred = predict(X, w, b)
plot_result(X, y, X, y_pred)

print(result_array)
