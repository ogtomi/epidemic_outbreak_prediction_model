import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime

from api_request import GoogleRequests
from filter_data import correlation_filter
from models import ewls, make_vector, stationary_ls, ls
from process_data import convert_to_weekly, get_data_for_comparison, get_mean_from_csv, predict_dataframe

plt.style.use('ggplot')

MODEL_ORDER = 3

def get_anchortime(get_time):
    if get_time == 0:
        return "2020-01-20 2022-02-28"

    if get_time == 1:
        last_date = datetime.strptime("2022-02-27" , "%Y-%m-%d").date()
        first_date = last_date - timedelta(days=MODEL_ORDER * 7 + 365)
        return str(first_date) + " " + str(last_date)
        # return "2021-02-28 2022-02-27"
    
    if get_time == 2:
        #"2021-01-01 2022-01-01"
        today_date = date.today()
        past_date = today_date - timedelta(days=365)
        anchor_time = str(past_date) + " " + str(today_date)
        return anchor_time
    
    if get_time == 3:
        start_date_csv = datetime.strptime("2020-01-20", "%Y-%m-%d").date()
        first_date = start_date_csv + timedelta(MODEL_ORDER * 7)
        last_date = first_date + timedelta(days=365)
        
        return str(first_date) + " " + str(last_date)

get_anchortime(3)
COUNTRY = "united states"
KEYWORDS = ['nausea', 'stomach pain']
CAT = '0'
TIMEFRAMES = ['today 12-m', 'today 3-m', 'today 1-m']
GPROP = ''
ANCHOR_TIME_MODEL = get_anchortime(3) 
ANCHOR_TIME_COMPARISON = get_anchortime(1)
ANCHOR_TIME_PREDICT = get_anchortime(2)
PATH_TO_CSV = 'covid_confirmed_usafacts.csv'
VALUE = 'covid_mean'

# google_requests = GoogleRequests(KEYWORDS, CAT, TIMEFRAMES, COUNTRY, GPROP, ANCHOR_TIME_MODEL)
# predict_requests = GoogleRequests(KEYWORDS, CAT, TIMEFRAMES, COUNTRY, GPROP, ANCHOR_TIME_PREDICT)
compare_requests = GoogleRequests(KEYWORDS, CAT, TIMEFRAMES, COUNTRY, GPROP, ANCHOR_TIME_COMPARISON)
model_requests = GoogleRequests(KEYWORDS, CAT, TIMEFRAMES, COUNTRY, GPROP, ANCHOR_TIME_MODEL)

# GETTING DATA FROM GOOGLE API AND PROCESSING IT
# model_data = google_requests.request_window()
# model_data = correlation_filter(model_data, KEYWORDS)

covid_data = get_mean_from_csv(PATH_TO_CSV)
weekly_covid_data = convert_to_weekly(covid_data)
weekly_covid_array = weekly_covid_data.to_numpy()
last_year_weekly_covid_data = get_data_for_comparison(weekly_covid_data, 0, MODEL_ORDER)
last_year_weekly_covid_data_array = last_year_weekly_covid_data.to_numpy()
# print(weekly_covid_array)
# frames = [model_data, weekly_covid_data]
# result_array = pd.concat(frames, axis=1)
# result_array = correlation_filter(result_array, [VALUE])

#FINAL WORD BANK
# word_bank = result_array.drop(VALUE, axis=1)
# word_bank_rows = len(word_bank.index)
# word_bank = word_bank.columns.values.tolist()

# BUILDING THE MODEL 
# X_predict = predict_requests.arrange_data(KEYWORDS) # ----> SWAP TO WORD_BANK
# X_compare = compare_requests.arrange_data(KEYWORDS)
X_model = model_requests.arrange_data(KEYWORDS)
# print(X_compare)
# X_compare.plot()

#vector_data = make_vector(X_predict)
print("VECTOR")
#vector_data_compare = make_vector(X_compare)
vector_data_model = make_vector(X_model)
print(len(vector_data_model), "vector compare")
print(len(last_year_weekly_covid_data), "weekly covid array")

#Y_predict = ewls(vector_data, len(X_predict.index), len(KEYWORDS), weekly_covid_array) # ----> COUNT ROWS AFTER
#Y_compare = ewls(vector_data_compare, len(X_compare.index), len(KEYWORDS), last_year_weekly_covid_data_array)

# realtime_mode True = realtime False = according to CSV
# Y_predict_dataframe = predict_dataframe(Y_predict, True)
# Y_compare_dataframe = predict_dataframe(Y_compare, False)
Y_model = ls(vector_data_model, len(X_model.index), len(KEYWORDS), last_year_weekly_covid_data_array)
Y_model_dataframe = predict_dataframe(Y_model, 2, MODEL_ORDER)

print("stationary ls")
# Y_stationary_ls = stationary_ls(vector_data_compare, len(X_compare.index), len(KEYWORDS), last_year_weekly_covid_data_array)
print("LS")
#Y_ls = ls(vector_data_compare, len(X_compare.index), len(KEYWORDS), last_year_weekly_covid_data_array)
#Y_stationary_ls_dataframe = predict_dataframe(Y_stationary_ls, False)
#Y_ls_dataframe = predict_dataframe(Y_ls, False)
# print(one_year_weekly_covid_data)
# print(len(one_year_weekly_covid_data))
# print(Y_compare_dataframe)
print(len(Y_model_dataframe.index))
print(len(last_year_weekly_covid_data_array))

plt.figure()
#plt.plot(Y_stationary_ls_dataframe)
#plt.plot(Y_compare_dataframe, label="Model data")
plt.plot(last_year_weekly_covid_data, label="Real cases")
plt.plot(weekly_covid_data)
plt.plot(Y_model_dataframe, label="LS")
plt.legend()
plt.show()