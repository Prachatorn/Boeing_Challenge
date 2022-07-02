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
test_pred = pd.read_csv("test_pred.csv")

# Removing rows where there are no values in a column.
train_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear", "Dealer_Listing_Price"]] = train_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear", "Dealer_Listing_Price"]].fillna(0)
test_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear"]]  = test_data[["SellerRating", "SellerRevCnt", "SellerZip", "VehListdays", "VehMileage", "VehYear"]].fillna(0)
train_data = train_data.fillna("DNE")
test_data = test_data.fillna("DNE")

# Seperating the independent and dependent variables in the training dataset.
X_train = train_data.drop("Dealer_Listing_Price", axis=1)
X_train = X_train.drop("Vehicle_Trim", axis=1)
y_train_trim = train_data["Vehicle_Trim"]

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

print("ok1")

logmodel = LogisticRegression(solver="liblinear")
logmodel.fit(train_data_new, y_train_trim)
y_trim_pred = logmodel.predict(test_data_new)
print(y_trim_pred)

data = {"ListingID": test_pred["ListingID"], "trim_pred": y_trim_pred, "price_pred": test_pred["price_pred"]}
df = pd.DataFrame(data)
df.to_csv("test_pred.csv")