# MachineLearningStocks in Python: A Starter Project and Guide

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

MachineLearningStocks is designed to be an **intuitive** and **highly extensible** template project applying machine learning to making stock predictions. My hope is that this project will help you understand the overall workflow of using machine learning to predict stock movements and also appreciate some of its subtleties. And of course, after following this guide and playing around with the project, you should definitely **make your own improvements** – if you're struggling to think of what to do, at the end of this readme I've included a long list of possibilities: take your pick.

Concretely, we will be cleaning and preparing a dataset of historical stock prices and fundamentals using `pandas`, after which we will apply a `scikit-learn` classifier to discover the relationship between stock fundamentals (e.g., PE ratio, debt/equity, float, etc.) and the subsequent annual price change (compared with an index). We then conduct a simple backtest, before generating predictions on current data.

## Contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Preliminaries](#preliminaries)
- [Historical Data](#historical-data)
  - [Historical Stock Fundamentals](#historical-stock-fundamentals)
  - [Historical Price Data](#historical-price-data)
- [Creating the Training Dataset](#creating-the-training-dataset)
  - [Preprocessing Historical Price Data](#preprocessing-historical-price-data)
  - [Features](#features)
    - [Valuation Measures](#valuation-measures)
    - [Financials](#financials)
    - [Trading Information](#trading-information)
  - [Parsing](#parsing)
- [Backtesting](#backtesting)
- [Current Fundamental Data](#current-fundamental-data)
- [Stock Prediction](#stock-prediction)
- [Unit Testing](#unit-testing)
- [Where to Go From Here](#where-to-go-from-here)
  - [Data Acquisition](#data-acquisition)
  - [Data Preprocessing](#data-preprocessing)
  - [Machine Learning](#machine-learning)
- [Contributing](#contributing)

## Overview

The overall workflow to use machine learning to make stocks prediction is as follows:

1. Acquire historical fundamental data – these are the _features_ or _predictors_.
2. Acquire historical stock price data – this is will make up the dependent variable, or label (what we are trying to predict).
3. Preprocess data.
4. Use a machine learning model to learn from the data.
5. Backtest the performance of the machine learning model.
6. Acquire current fundamental data.
7. Generate predictions from current fundamental data.

This is a very generalized overview, but in principle, this is all you need to build a fundamentals-based ML stock predictor.

This project uses pandas-datareader to download historical price data from Yahoo Finance. However, in the past few weeks this has become extremely inconsistent – it seems like Yahoo have added some measures to prevent the bulk download of their data. I will try to add a fix, but for now, take note that `download_historical_prices.py` may be deprecated.

As a temporary solution, I've uploaded `stock_prices.csv` and `sp500_index.csv`, so the rest of the project can still function.

I expect that after so much time there will be many data issues. To that end, I have decided to upload the other CSV files: `keystats.csv` (the output of `parsing_keystats.py`) and `forward_sample.csv` (the output of `current_data.py`).

## Quickstart

If you want to throw away the instruction manual and play immediately, clone this project, then download and unzip the data file into the same directory. Then, open an instance of terminal and cd to the project's file path, e.g.

```bash
cd Users/User/Desktop/MachineLearningStocks
Then, run the following in terminal:

pip install -r requirements.txt
python download_historical_prices.py
python parsing_keystats.py
python backtesting.py
python current_data.py
pytest -v
python stock_prediction.py

Otherwise, follow the step-by-step guide below.

Preliminaries
This project uses python 3.6, and the common data science libraries pandas and scikit-learn. If you are on python 3.x less than 3.6, you will find some syntax errors wherever f-strings have been used for string formatting. These are fortunately very easy to fix (just rebuild the string using your preferred method), but I do encourage you to upgrade to 3.6 to enjoy the elegance of f-strings. A full list of requirements is included in the requirements.txt file. To install all of the requirements at once, run the following code in terminal:

pip install -r requirements.txt

To get started, clone this project and unzip it. This folder will become our working directory, so make sure you cd your terminal instance into this directory.

Historical Data
Data acquisition and preprocessing is probably the hardest part of most machine learning projects. But it is a necessary evil, so it's best to not fret and just carry on.

For this project, we need three datasets:

Historical stock fundamentals.
Historical stock prices.
Historical S&P500 prices.
We need the S&P500 index prices as a benchmark: a 5% stock growth does not mean much if the S&P500 grew 10% in that time period, so all stock returns must be compared to those of the index.

Historical Stock Fundamentals
Historical fundamental data is actually very difficult to find (for free, at least). Although sites like Quandl do have datasets available, you often have to pay a pretty steep fee.

It turns out that there is a way to parse this data, for free, from Yahoo Finance. On his page you will be able to find a file called intraQuarter.zip, which you should download, unzip, and place in your working directory. Relevant to this project is the subfolder called _KeyStats, which contains HTML files that hold stock fundamentals for all stocks in the S&P500 between 2003 and 2013, sorted by stock. However, at this stage, the data is unusable – we will have to parse it into a nice CSV file before we can do any ML.

Historical Price Data
In the first iteration of the project, I used pandas-datareader, an extremely convenient library which can load stock data straight into pandas. However, after Yahoo Finance changed their UI, datareader no longer worked, so I switched to Quandl, which has free stock price data for a few tickers, and a python API. However, as pandas-datareader has been fixed, we will use that instead.

Likewise, we can easily use pandas-datareader to access data for the SPY ticker. Failing that, one could manually download it from yahoo finance, place it into the project directory and rename it sp500_index.csv.

The code for downloading historical price data can be run by entering the following into terminal:

python download_historical_prices.py


Creating the Training Dataset
Our ultimate goal for the training data is to have a 'snapshot' of a particular stock's fundamentals at a particular time, and the corresponding subsequent annual performance of the stock.

For example, if our 'snapshot' consists of all of the fundamental data for AAPL on the date 28/1/2005, then we also need to know the percentage price change of AAPL between 28/1/05 and 28/1/06. Thus our algorithm can learn how the fundamentals impact the annual change in the stock price.

In fact, this is a slight oversimplification. In fact, what the algorithm will eventually learn is how fundamentals impact the outperformance of a stock relative to the S&P500 index. This is why we also need index data.

Preprocessing Historical Price Data
When pandas-datareader downloads stock price data, it does not include rows for weekends and public holidays (when the market is closed).

However, referring to the example of AAPL above, if our snapshot includes fundamental data for 28/1/05 and we want to see the change in price a year later, we will get the nasty surprise that 28/1/2006 is a Saturday. Does this mean that we have to discard this snapshot?

By no means – data is too valuable to callously toss away. As a workaround, I instead decided to 'fill forward' the missing data, i.e we will assume that the stock price on Saturday 28/1/2006 is equal to the stock price on Friday 27/1/2006.

Features
Below is a list of some of the interesting variables that are available on Yahoo Finance.

Valuation Measures
Market Cap
Enterprise Value
```
