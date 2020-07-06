import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model


def main():
    t_start = time.time()
    
    # Load and clean data
    data = pd.read_csv('elasticsearch-seed.csv', header = 0)
    
    data = data[['BathsTotal',                     # restrict analysis to select features
                'BedsTotal', 
                'CDOM', 
                'LotSizeAreaSQFT', 
                'SqFtTotal', 
                'ElementarySchoolName', 
                'ClosePrice']]

    data = data[data.ClosePrice.notna()] # remove rows without a label

    # Convert categorical variable into numerical value (via one-hot encoding)
    home_data = data.copy()
    home_data = pd.get_dummies(home_data, prefix_sep='_', drop_first=True)
    
    # Log transform response
    home_data['log_ClosePrice'] = np.log(home_data['ClosePrice'])
    
    # Split data into train and test sets
    X = home_data.drop(['ClosePrice', 'log_ClosePrice'], axis=1)
    Y = home_data['log_ClosePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Build model
    line = linear_model.LinearRegression(normalize=True)
    line.fit(X_train, y_train)

    # Predict home values on test set
    ytest = np.array(y_test)
    y_pred = line.predict(X_test)

    # Print metrics
    # print('Coefficients: \n', model_1.coef_)
    print('Mean squared error: %.2f' % mean_squared_error(ytest, y_pred))
    print('R-squared: %.2f' % r2_score(ytest, y_pred))
    
    t_end = time.time()
    print("time seconds: %f" %(time.time() - t_start))
    
if __name__ == '__main__':main()