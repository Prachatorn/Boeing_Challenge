import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import scale

# Read in both the training and test dataset. 
train_data = pd.read_csv("training_DataSet.csv", index_col=0)
test_data = pd.read_csv("test_Dataset.csv", index_col=0)

# Removing rows where there are no values in a column.
test_data = test_data.dropna() # Removed 1053 rows
train_data = train_data.dropna() # Removed 190 rows

# Seperating the independent and dependent variables in the training dataset.
X_train = train_data.drop("Dealer_Listing_Price", axis=1)
X_train = X_train.drop("Vehicle_Trim", axis=1)
y_train_price = train_data["Dealer_Listing_Price"]

car_data = [X_train, test_data]
total_data = pd.concat(car_data)

# Creating dummies variables in both the training and test dataset for categorical variables. 
total_data_new = pd.get_dummies(total_data, columns=["SellerCity", "SellerIsPriv", "SellerListSrc", "SellerName",
                                                    "SellerState", "VehBodystyle", "VehCertified", "VehColorExt",
                                                    "VehColorInt", "VehDriveTrain", "VehEngine", "VehFeats",
                                                    "VehFuel", "VehHistory", "VehMake", "VehModel", "VehPriceLabel", 
                                                    "VehSellerNotes", "VehType", "VehTransmission"])

train_data_new = total_data_new.iloc[:5045,:]
test_data_new = total_data_new.iloc[5045:, :]

# Predicting the values of the prices of cars using Linear Regression
lm = LinearRegression()
lm.fit(train_data_new, y_train_price)
y_pred_train = lm.predict(train_data_new)
y_pred_test = lm.predict(test_data_new)
#print(y_pred)
lr_train_mse = mean_squared_error(y_train_price, y_pred_train)
print(lr_train_mse)
lr_train_r2 = r2_score(y_train_price, y_pred_train)
print(lr_train_r2)

#rf = RandomForestRegressor(max_depth=2, random_state=42)
#rf.fit(X_train, y_train)