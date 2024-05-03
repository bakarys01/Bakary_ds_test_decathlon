import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import datetime
import pickle
import datetime

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')



def handle_outliers(df):
    Q1 = df['turnover'].quantile(0.25)
    Q3 = df['turnover'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    min_non_outliers = df['turnover'][df['turnover'] >= lower_bound].min()
    max_non_outliers = df['turnover'][df['turnover'] <= upper_bound].max()

    df['turnover'] = np.where(df['turnover'] < lower_bound, min_non_outliers, 
                               np.where(df['turnover'] > upper_bound, max_non_outliers, df['turnover']))
    return df

def get_time_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df
    
def add_lags(df):
    target_map = df['turnover'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('30 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('60 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('90 days')).map(target_map)
    return df

class MultiStepForecasting:
    def __init__(self):
        self.bu_feat_file = 'data/bu_feat.csv.gz'
        self.train_file = 'data/train.csv.gz'
        #self.test_file = 'data/test.csv.gz'
        self.model = None
        self.equals_line = "="*20

    def load_data(self):
        
        print("".join([self.equals_line, "Loading data....", self.equals_line]))
        
        self.bu_feat = pd.read_csv(self.bu_feat_file, compression='gzip')
        self.train = pd.read_csv(self.train_file, compression='gzip')
        
        
        print(" Data loaded !")
        
    def preprocess(self):
        
        print("".join([self.equals_line, "Data Processing....", self.equals_line]))
        
        # Merge datasets on 'but_num_business_unit'
        self.data = pd.merge(self.train, self.bu_feat, on='but_num_business_unit')
        
        # convert day_id to datetime and make sure that data is ordered in time series order
        self.data['day_id'] = pd.to_datetime(self.data['day_id'])
        self.data.sort_values('day_id', inplace=True)
        self.data = self.data.set_index('day_id')
        self.data.index = pd.to_datetime(self.data.index)

        # feature engineering
        self.data = get_time_features(self.data)

        # remove negative turnover and error (turnover >= 1000 000)
        self.data = self.data[(self.data['turnover']>=0 ) & (self.data['turnover']<1000000)]
        
        # handle outliers in turnover column
        #self.data = handle_outliers(self.data)

        # handle outliers at week level
        self.data = self.data.groupby('weekofyear').apply(handle_outliers)
        self.data = self.data.reset_index(level=0, drop=True)

        #self.data = add_lags(self.data)

        
        df = self.data.copy()
        # Split data into training and validation sets
        # Determine the cutoff date by subtracting 8 weeks from the maximum date in the dataset
        max_date = df.index.max()
        cutoff_date = max_date - datetime.timedelta(weeks=8)
        
        #self.train_data = self.data.copy()
        self.train_data = df[df.index < cutoff_date]
        self.val_data = df[df.index >= cutoff_date]
        
        # features selection
        self.features = ['but_num_business_unit', 'dpt_num_department', 'but_latitude', 'but_longitude','but_region_idr_region', 'zod_idr_zone_dgr', 
       'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'dayofweek']
        
        self.target = 'turnover'
        
        self.X_train = self.train_data[self.features]
        self.X_val = self.val_data[self.features]
        self.y_train = self.train_data[self.target]
        self.y_val = self.val_data[self.target]
        
        print(" Data Processed !")

    def train_model(self):
        
        print("".join([self.equals_line, "Model Training....", self.equals_line]))  
        
        # Train XGBoost model 
        self.model = xgb.XGBRegressor(objective ='reg:squarederror', random_state=2024, 
                                      colsample_bytree = 0.35, 
                                      n_estimators=2000, #learning_rate=0.05,
                                      early_stopping_rounds=200)
        
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)], verbose=100)

        print(" Done !")
     

    def evaluate(self):
        # Evaluate model using RMSE
        self.y_val_pred = self.model.predict(self.X_val)
        self.y_val_pred = np.maximum(self.y_val_pred, 0)

        self.y_train_pred = self.model.predict(self.X_train)
        self.y_train_pred = np.maximum(self.y_train_pred, 0)
        rmse_train = mean_squared_error(self.y_train, self.y_train_pred, squared=False) # False for root calculation
        
        rmse = mean_squared_error(self.y_val, self.y_val_pred, squared=False) # False for root calculation
        
        print("".join([self.equals_line, "Model Evaluation", self.equals_line]))
        
        print(f'Training RMSE: {rmse_train}')
        
        print("".join([self.equals_line, "=", self.equals_line]))
        
        print(f'Validation RMSE: {rmse}')
        print("".join([self.equals_line, "=", self.equals_line]))

    def save_model(self):
        # save the model
        pickle.dump(self.model, open('model_artefacts/xg_regressor.pkl', "wb"))
        print("".join([self.equals_line, "=", self.equals_line, "  Model Saved successfully!"]))
        
    def get_feature_importance(self):
        # visualize xgboost feature importance
        fi = pd.DataFrame(data=self.model.feature_importances_,
                     index=self.model.feature_names_in_,
                     columns=['importance'])
        fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
        plt.show()

    def get_model(self):

        model = pickle.load(open('model_artefacts/xg_regressor.pkl', "rb"))
        return model 
        
    def preprocess_test_data(self, test_file_path):

        self.test = pd.read_csv(test_file_path, compression='gzip')
        
        self.test = pd.merge(self.bu_feat, self.test, on='but_num_business_unit')
        # convert day_id to datetime and make sure that data is ordered in time series order
        self.test['day_id'] = pd.to_datetime(self.test['day_id'])
        self.test.sort_values('day_id', inplace=True)
        self.test = self.test.set_index('day_id')
        self.test.index = pd.to_datetime(self.test.index)

        self.test = get_time_features(self.test)
        self.test = self.test[self.features]

        return self.test
        
    def predict(self, preprocessed_test):
        
        #load the model
        self.model = pickle.load(open('model_artefacts/xg_regressor.pkl', "rb"))
        
        # Predict on unseen data
        self.test_pred = self.model.predict(preprocessed_test)
        self.test_pred = np.maximum(self.test_pred, 0)

        return self.test_pred
        


    def run(self):
        self.load_data()
        self.preprocess()
        self.train_model()
        self.evaluate()
        self.save_model()
        
        #self.predict()
        #test_file_path = 'data/test.csv.gz'
        #self.preprocess_test_data(test_file_path)
        #self.predict(self.test)


if __name__ == "__main__":
    forecasting = MultiStepForecasting()
    forecasting.run()
