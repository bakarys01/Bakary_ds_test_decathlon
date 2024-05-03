# Turnover Forecasting Technical Test

This project aims to forecast the turnover for the next 8 weeks at the store-department level to assist store managers in making mid-term decisions driven by economic data.

## Project Structure

The solution consists of the following files:

- `EDA.ipynb`: This Jupyter notebook contains the exploratory data analysis and answers to preliminary questions about the data. The questions include identifying the department with the highest turnover in 2016, the top 5 week numbers for department 88 in 2015 in terms of turnover over all stores, the top performer store in 2014, and guessing the kind of sport that represents departments 73 and 117 based on sales. It also includes additional insights drawn from the data with relevant plots and figures.

- `turnover_forecaster.py`: This Python script contains the end-to-end pipeline for the forecasting task. It includes data preprocessing, model training, evaluation, and forecasting.

- `ML.ipynb`: This Jupyter notebook contains the execution of the pipeline, analysis of the pipeline model, final prediction on the test data, choice of evaluation metric (RMSE), and next steps for improving the pipeline.

- `requirements.txt`: This file lists the Python libraries used in the project.

