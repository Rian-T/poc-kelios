from time import sleep
from forecasting import predict_for_day
import matplotlib.pyplot as plt
import streamlit as st
import datetime

st.title("Kelios challenge - Forecasting POC")
st.markdown('### Team 3 - Mooving Partners')
st.text("This is our live demo for the Kelios Data challenge.")

d = st.date_input(
     "Date from which to do the forecasting",
     datetime.date(2021, 12, 6))
st.write('Selected date is:', d)

# options = st.selectbox(
#      'What are your favorite colors',
#      ('00:00', '04:00', '08:00', '12:00'))

# st.write('You selected:', options)
# t = datetime.datetime.strptime(options, '%H:%M').time()
t = datetime.time(0,0)

date = datetime.datetime.combine(d, t)
datetime_pred = date.strftime("%Y-%m-%d %H:%M:%S")+"+00:00"
fig, ax = plt.subplots()
plt.ylabel('Nb of perturbations')
day = d.strftime("%c")[:11]
plt.title(f"Forecast for the next 12 hours\n for {day}")
preds, raw_preds = predict_for_day(date.strftime("%Y-%m-%d %H:%M:%S")+"+00:00", ax)
st.pyplot(fig)
plt.show()