import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    train = pd.read_csv("linreg_train.csv", header=None)
	train = train.transpose()
	X = train.iloc[1:,0].values
	y = train.iloc[1:,1].values
	X = X.reshape(-1,1)
	y = y.reshape(-1,1)


    from sklearn.preprocessing import PolynomialFeatures
	poly_reg = PolynomialFeatures(degree = 2)
	X_poly = poly_reg.fit_transform(X)
	poly_reg.fit(X_poly, y)
	from sklearn.linear_model import LinearRegression
	regressor = LinearRegression()
	regressor.fit(X_poly, y)
	X_poly = poly_reg.fit_transform(X)
	y_pred = regressor.predict(X_poly)
	return y_pred


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
