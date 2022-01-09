import streamlit as st
from plotly import graph_objs as go
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from datetime import date,datetime,timedelta
import pandas as pd
import time
import math
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf

st.title("Stock Price Prediction")
st.warning("Model's accuracy varies from Stock to Stock")
st.markdown('#### Enter the stock ticker(for NSE stocks add .NS after the name)')
stock_name = st.text_input("")

start_date = "2012-01-01"
end_date = date.today().strftime("%Y-%m-%d")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache
def load_data(stock,start,end):
    yf.pdr_override()
    data = pdr.get_data_yahoo({stock}, start=start, end=end)
    data.reset_index(inplace=True)
    return data


st.markdown("#### Select the date on which you want to predict the closing price")
pred_date = st.date_input("")
def load_new_data(stock,pred_date):
  pred_date -= timedelta(days=1)
  end_d = pred_date.strftime("%Y-%m-%d")
  new_data = pdr.get_data_yahoo({stock_name}, start="2010-01-01", end=end_d)
  new_data.reset_index(inplace=True)
  return new_data
submit = st.button("Predict")


# CREATING DICTS FOR JUMP IN TIME GRAPH
month_dict = dict(count=1,label="1M",step="month",stepmode="backward")
six_step = dict(count=6,label="6M",step="month",stepmode="backward")
ytd_dict = dict(count=1,label="YTD",step="year",stepmode="todate")
year1_dict = dict(count=1,label="1Y",step="year",stepmode="backward")
all_dict = dict(label='MAX',step="all")

if(len(stock_name)!=0):
  data_load_state = st.text("Loading Data ...")
  df = load_data(stock_name,start_date,end_date)
  data_load_state.text("Loading Data ... Done!")

  data = df.filter(['Close','Date'])
  data_np = data['Close'].values
  training_data_len = math.ceil(len(data_np)*0.8)
  test_data_len = len(data_np) - training_data_len

  # SCALING THE DATA
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data_np.reshape(-1,1))
  train_data = scaled_data[:training_data_len]
  test_data = scaled_data[training_data_len-60:]

  # SPLITTING INTO X_TRAIN,X_TEST,Y_TRAIN,Y_TEST
  x_train = []
  y_train = []
  x_test=[]
  y_test=data_np[training_data_len:]
  for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
  x_train, y_train = np.array(x_train), np.array(y_train)

  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])
  x_test = np.array(x_test)

  # RESHAPING THE DATA
  x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
  x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

  if submit:
    model = load_model('keras_model.h5')

    preds = model.predict(x_test)
    predictions = scaler.inverse_transform(preds)

    mse = math.sqrt(mean_squared_error(predictions, y_test))

    st.text(f"The mean squared error of the model is {mse}")

    # NEW DF CREATED FOR VISUALISATION
    train_part = data[:training_data_len]
    valid = data[training_data_len:]
    valid["Prediction"] = predictions

    # GRAPH AFTER THE MODEL
    def plot_pred_data(stock_name):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_part['Date'], y=train_part['Close'],mode='lines',name='Train'))
        fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Close'],mode='lines',name='Actual'))
        fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Prediction'],mode='lines',name='Prediction'))
        fig.layout.update(title_text=f"{stock_name} Price Model")
        fig.update_layout(xaxis=dict(rangeselector=dict(buttons=list([month_dict,six_step,ytd_dict,year1_dict,all_dict])),rangeslider=dict(visible=True),type="date"))
        fig.update_layout(yaxis_title='Price')
        st.plotly_chart(fig)

    plot_pred_data(stock_name)

    new_df = load_new_data(stock_name,pred_date)
    data_2 = df.filter(['Close','Date'])
    data_2_close = data_2['Close']
    # LAST 60 DAYS DATA
    last_60 = data_2_close[-60:].values
    scaled_last60  = scaler.transform(last_60.reshape(-1,1))
    # CREATING EMPTY DATASET
    x_test_60=[]
    x_test_60.append(scaled_last60)
    x_test_60 = np.array(x_test_60)
    x_test_60 = np.reshape(x_test_60,(x_test_60.shape[0],x_test_60.shape[1],1))

    
    def get_price():
      pred_1 = model.predict(x_test_60)
      predictions_1 = scaler.inverse_transform(pred_1)
      answer = st.markdown(f"#### The closing price on {pred_date} will be {predictions_1[0][0]}")

    with st.spinner('Predicting the price . . .'):
      get_price()

    st.success("Done!")

    
