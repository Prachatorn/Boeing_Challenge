import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Read in both the training and test dataset. 
train_data = pd.read_csv("training_DataSet.csv")
test_data = pd.read_csv("test_Dataset.csv")

# Filling the missing entries with 0 for numerical variables and DNE for categorical variables.
train_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear", "Dealer_Listing_Price"]] = train_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear", "Dealer_Listing_Price"]].fillna(0)
test_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear"]]  = test_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear"]].fillna(0)
train_data = train_data.fillna("DNE")
test_data = test_data.fillna("DNE")

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

train_data_new = total_data_new.iloc[:6298,:]
test_data_new = total_data_new.iloc[6298:, :]

# Predicting the values of the prices of cars using Linear Regression
#lm = LinearRegression()
#lm.fit(train_data_new, y_train_price)
#y_pred_train = lm.predict(train_data_new)
#y_pred_test = lm.predict(test_data_new)

# Finding the Mean Squared Error and R2 Score.
#lr_train_mse = mean_squared_error(y_train_price, y_pred_train) # 76854.01502679026
#print(lr_train_mse)
#lr_train_r2 = r2_score(y_train_price, y_pred_train) # 0.9986754841700016
#print(lr_train_r2)

# Predicting the values of the prices of cars using Random Forest Regressor
rf = RandomForestRegressor(max_depth=2, random_state=42)
rf.fit(train_data_new, y_train_price)

y_rf_train_pred = rf.predict(train_data_new)
y_rf_test_pred = rf.predict(test_data_new)
print(y_rf_test_pred)

# Finding the Mean Squared Error and R2 Score.
rf_train_mse = mean_squared_error(y_train_price, y_rf_train_pred) # 40252186.81595471
print(rf_train_mse)
rf_train_r2 = r2_score(y_train_price, y_rf_train_pred) # 0.3795209511339379
print(rf_train_r2)

data = {"ListingID": test_data["ListingID"], "price_pred": y_rf_test_pred}
df = pd.DataFrame(data)
df.to_csv("test_pred.csv")