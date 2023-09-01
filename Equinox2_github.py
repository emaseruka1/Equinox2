# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 19:37:25 2023

@author: Emmanuel Maseruka
"""
from requests import Request, Session
import json
from binance.client import Client
import talib
import datetime
import requests
from etherscan import Etherscan
import numpy as np
import sqlite3
import pandas as pd
from coinmarketcapapi import CoinMarketCapAPI
import pprint
from pytrends.request import TrendReq
import sys
import time
import smtplib
from email.message import EmailMessage


# Email configuration
sender_email = ''
receiver_email = ''
subject = 'EQUINOX 2 UPDATE'


smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = ''
smtp_password = ''




def animated_loading():
    chars = "/—\|"  # Characters for animation
    for char in chars:
        sys.stdout.write(f"\rLoading... {char}")
        sys.stdout.flush()
        time.sleep(0.2)  # Adjust the delay for animation speed

def clear_screen():
    print("\033[H\033[J", end='')

import os
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_robot():
    print("  _____ ")
    print(" /|o o\\")
    print("(|  ^  |)")
    print(" | (_) |")
    print(" |_____|")

def animate_robot():
    for _ in range(5):
        clear_screen()
        draw_robot()
        time.sleep(0.5)
        clear_screen()
        time.sleep(0.5)

# Call the animation function
animate_robot()

def print_red(text):
    print("\033[91m" + text + "\033[0m")

print("EQUINOX 2")

for _ in range(5):
    animated_loading()

print("\nProcess complete!")

#set time frames
today=datetime.date.today()
yesterday_start = datetime.date.today() - datetime.timedelta(days=1)
yesterday_start = datetime.datetime.combine(yesterday_start, datetime.time.min)
yesterday_stop= yesterday_start.replace(hour=23, minute=55)
yesterday_start = yesterday_start.strftime('%Y-%m-%d %H:%M:%S')
yesterday_stop = yesterday_stop.strftime('%Y-%m-%d %H:%M:%S')

print("connecting to SQlite Database")
#setup databases
connection = sqlite3.connect("equinox2.db")



#api access
etherscan_api = ""
binance_key= ''
binance_secret= ''
coinmarketcap_key=""

#the below tickers should be related
binance_asset_ticker=["ETHBUSD"]
coinmarketcap_ticker=["ETH"]

#DL tables for each asset

def table_exists(table_name, db_connection):
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    result = db_connection.execute(query)
    return result.fetchone() is not None

column_definitions = [
    'Volume REAL',#coinmarketcap
    'sma14 REAL',#coinmarketcap
    'ema REAL',#coinmarketcap
    'macd_signal REAL',#coinmarketcap
    'macd REAL',#coinmarketcap
    'Close REAL',#coinmarketcap
    'Close_btc REAL',#coinmarketcap
    #'transaction_growth REAL',#cant get
    #'block_size_bytes REAL',#cant get***************
    'google_trend REAL',#google
    'gas_price_wei REAL', #etherscan
    #'token_tr REAL',#cant get***************
    #'compound_score REAL'#cant get twitter***************
]

#ML tables for each asset
ML_column_definitions = ['LSTM REAL','GRU REAL','TCN REAL','output']


print("Fetching Ethereum Blockchain Data from Etherscan.io ...")

for _ in range(5):
    animated_loading()

print("\nProcess complete!")

#fetch data from etherscan
etherscan_client = Etherscan(etherscan_api)
gas = etherscan_client.get_gas_oracle()
gas =np.mean((int(gas['SafeGasPrice']), int(gas['ProposeGasPrice']) , int(gas['FastGasPrice'])))


#fetch data from binance
binance_client = Client(binance_key,binance_secret)

#fetch coinmarketcap data
coinmarketcap_url ="https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"

parameters={
    'symbol':'BTC',
    'convert':'USD',
    }
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY':coinmarketcap_key
    
    }
session = Session()
session.headers.update(headers)
response=session.get(coinmarketcap_url,params=parameters)

print("Fetching Data from Coinmarketcap...")
btc_price = pprint.pprint(json.loads(response.text)['data']['BTC'][0]['quote']['USD']['price'])


for _ in range(5):
    animated_loading()

print("\nProcess complete!")

print("Fetching Data from Google Trends...")
#fetch data from google trends
pytrends = TrendReq(hl='en-US', tz=360)
kw_list = ["Ethereum","Crypto","Cryptocurrency","Eth"]
pytrends.build_payload(kw_list, cat=0, timeframe='now 1-d', geo='', gprop='')
gt = pytrends.interest_over_time()

gt=gt.reset_index()
gt['date'] = pd.to_datetime(gt['date'])
gt['date'] = gt['date'].dt.date
gt=gt[gt['date'] == today]

gt = gt.drop(columns=['isPartial', 'date'])

gt = gt.mean().mean()





for i in binance_asset_ticker:
    
    #print("Connecting to relevant SQL tables...")
# Connect to the SQLite database
    conn = sqlite3.connect('equinox2.db')

# Check if the features input table exists
    if table_exists(i, conn):
        #print("Database Table exists.")
        xu=1
    else:
        create_new_table_query = f"CREATE TABLE {i}  ({', '.join(column_definitions)});"
        conn.execute(create_new_table_query)
        
#get coinmarketcap price, volume data and btc price data into the table
    

    #print("Fetching Coinmarketcap data")
    
    
    
    parameters={'symbol':'ETH','convert':'USD'}
    response=session.get(coinmarketcap_url,params=parameters)
    price = pprint.pprint(json.loads(response.text)['data']['ETH'][0]['quote']['USD']['price']) #today's price
    
    
# retrieve yesterday price and calculate return
    query = "SELECT price FROM ETH_bubble ORDER BY rowid DESC LIMIT 1"
    connection.execute(query)
    #yesterday_price = connection.fetchone()
    yesterday_price=1600
    price=1650
    today_return=(price/yesterday_price)-1
    
#update bubble table
    
    bubble_table_data=[(334,price,today_return)]
    insert_query = "INSERT INTO ETH_bubble VALUES (?,?,?)" 
    #connection.execute(insert_query,bubble_table_data)
    #connection.commit()  

    

    volume = pprint.pprint(json.loads(response.text)['data']['ETH'][0]['quote']['USD']['volume_24h'])
    
    
#fetch data from bubble table to calculate technical indicators


    print("Processing Input Features")
    


    data = pd.read_sql("SELECT price FROM ETH_bubble", connection) 
    
    sma14=data['price'].rolling(14).mean()
    sma14=sma14.iloc[-1]
    
    
    ema = talib.EMA(data['price'], timeperiod=14)
    ema = ema.iloc[-1]

    macd, macd_signal, _ = talib.MACD(data['price'], fastperiod=5,slowperiod=10, signalperiod=5)
    macd=macd.iloc[-1]
    macd_signal=macd_signal.iloc[-1]
    
    today_DL_features_data = [(volume, sma14,ema,macd_signal,macd,price,btc_price,gt,gas,1,1,1,1)]

#insert all feature values for today into the features table
    #insert_query = f"INSERT INTO {i} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)" 
    #connection.execute(insert_query,today_DL_features_data)
    #connection.commit()  


    print("Running Deep Learning Models")
    for _ in range(5):
        animated_loading()

    print("\nProcess complete!")


#run the saved DL algos




   
#send their predictions to ML table overtrading algo

# Check if the features input table exists

    ml_table_name = "ML_" + str(i)
    if table_exists(ml_table_name, conn):
        print("Table exists.")
    else:
        create_new_table_query = f"CREATE TABLE {ml_table_name}  ({', '.join(ML_column_definitions)});"
        conn.execute(create_new_table_query)    

    DL_return_prediction = [(0.03,0.04,0.01)] ### testing data!!! Not actual Algo
    
    #input those DL predictions into ML algo
    
    
    
    
    
    print("Detecting Overtrade Signals: Machine Learning loading...")
    #add saved ML algo here and produce prediction
    
    ML_output = 1###  testing data!!! Not actual Algo
    
    
    for _ in range(5):
        animated_loading()

    print("\nProcess complete!")
    
    
    today_ML_features_data =[(DL_return_prediction),ML_output]
    
    
#insert all ML values for today into the ML table
    insert_query = f"INSERT INTO {ml_table_name} VALUES (?,?,?,?)" 
    #connection.execute(insert_query,today_ML_features_data)
    #connection.commit()      
    
    
    trade_list=[]
#if ML output is 1, see Asset bubble check table and if 1 (bubble exists), check diebold spillover table for emitter or gainer decision. if gainer dont trade that asset
    if ML_output==1:
        
        print("No Overtrade signal")
        
        trade_list.append(i)
        
        #query = "SELECT ongoing FROM ETH_bubble_check ORDER BY rowid DESC LIMIT 1"
        #connection.execute(query)
        bubble_exits=1
        #bubble_exits = connection.fetchone()
        
        print("Check for Market Bubbles")
        
        for _ in range(5):
            animated_loading()

        print("\nProcess complete!")
        
        if bubble_exits==1:
            
            print("WARNING: Market is in Bubble. Check Diebold SpillOver Effects!!!")
            
            for _ in range(5):
                animated_loading()

            print("\nProcess complete!")
            
            query = "SELECT contribution FROM ETH_bubble_check where asset=={i}"
            #connection.execute(query)
            #contribution = connection.fetchone()
            contribution ="else"
            if contribution=="gainer":
                
                print("Asset is a net gainer. Drop from trade list!")
                trade_list.pop(i)
                
            else:
                print("Asset is a net emitter. Keep on trade list!")
        
        else:
            print("No Bubble detected. Keep trade list as is!")
                    
    else:
        print_red("Overtrade Alert. Halting Trade")
        
        for _ in range(5):
            animated_loading()

        print("\nProcess complete!")

        sys.exit()
    
        
    #connect to binance
    
    print("Connecting to Binance")
    
    for _ in range(5):
        animated_loading()

    print("\nProcess complete!")
    
    binance_client = Client(binance_key,binance_secret)
    
    avail_balance = binance_client.get_asset_balance(asset='BUSD') 
    avail_balance=52
    if avail_balance >50:
        
        print("Uninvested money available. Advance to Trade Execution")
        
        weights=avail_balance/len(trade_list)
        
        #assign Markowitz weights
        
        print("Executing Markowitz Portfolio Optimization")
        
        price_key = f"https://api.binance.com/api/v3/ticker/price?symbol={i}"
        curr_price = requests.get(price_key)  
        curr_price= curr_price.json()

        curr_price=float(curr_price['price'])
        
        buy_quantity = weights/curr_price
        #binance_client.create_test_order(symbol=i,side='BUY',type ='LIMIT',price =curr_price ,quantity = round(buy_quantity,5),timeInForce='GTC')
        print("Trade test order sent to Binance!!!")
        
        body = 'Equinox traded today'

# Create the email content
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
        
        print("Email Update Sent!")
        
        
    else: 
        
        print("Account Empty. Fund Binance Account")
        
        body = 'There have been no trades today'

# Create the email content
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
        
        print("Email Update Sent!")
        
        print("Logging off!")
        
        for _ in range(5):
            animated_loading()

        print("\nProcess complete!")
    














