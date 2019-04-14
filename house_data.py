import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.stats import zscore

house_data = pd.read_csv('kc_house_data.csv')
corr = house_data[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'grade', 'sqft_above', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_basement']].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True)
plt.show()

# Remove outliers
house_data['price_zscore'] = zscore(house_data['price'])
house_data['is_outlier'] = house_data['price_zscore'].apply(
  lambda x: x <= -5 or x >= 5
)
house_data.drop(index=house_data[house_data['is_outlier']].index, inplace=True)

model = LinearRegression(fit_intercept=True)

# Linear regression

x = house_data['sqft_living']
y = house_data['price']

model.fit(x[:, np.newaxis], y)
print(model.intercept_)
print(model.coef_)

xfit = np.linspace(0, 15000, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit, 'b')
plt.title('Linear regression for price against sqft_living (all houses)')
plt.show()

x = house_data[house_data['yr_renovated'] == 0]['sqft_living']
y = house_data[house_data['yr_renovated'] == 0]['price']

model.fit(x[:, np.newaxis], y)
print(model.intercept_)
print(model.coef_)

xfit = np.linspace(0, 15000, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit, 'b')
plt.title('Linear regression for price against sqft_living (houses without renovation)')
plt.show()

x = house_data[house_data['yr_renovated'] != 0]['sqft_living']
y = house_data[house_data['yr_renovated'] != 0]['price']

model.fit(x[:, np.newaxis], y)
print(model.intercept_)
print(model.coef_)

xfit = np.linspace(0, 15000, 10000)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit, 'b')
plt.title('Linear regression for price against sqft_living (houses with renovation)')
plt.show()

# Polynomial regression
x = house_data[house_data['yr_renovated'] == 0]['grade']
y = house_data[house_data['yr_renovated'] == 0]['price']
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(x[:, np.newaxis])
plt.scatter(x, y)
plt.plot(x, yfit, 'b')
plt.title('Polynomial regression for price against grade (houses with renovation)')
plt.show()

x = house_data[house_data['yr_renovated'] != 0]['grade']
y = house_data[house_data['yr_renovated'] != 0]['price']
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(x[:, np.newaxis])
plt.scatter(x, y)
plt.plot(x, yfit, 'b')
plt.title('Polynomial regression for price against grade (houses with renovation)')
plt.show()

class GaussianFeatures(BaseEstimator, TransformerMixin):
	def __init__(self, N, width_factor=2.0):
		self.N = N
		self.width_factor = width_factor
	@staticmethod
	def _gauss_basis(x, y, width, axis=None):
		arg = (x - y)/width
		return np.exp(-0.5 * np.sum(arg**2, axis))

	def fit(self, X, y=None):
		self.centers_ = np.linspace(X.min(), X.max(), self.N)
		self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
		return self

	def transform(self, X):
		return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)

# Lasso regression
x = house_data[house_data['yr_renovated'] == 0]['grade']
y = house_data[house_data['yr_renovated'] == 0]['price']
gauss_model = make_pipeline(GaussianFeatures(5), Lasso(alpha=0.1))
gauss_model.fit(x[:, np.newaxis], y)

yfit = gauss_model.predict(x[:, np.newaxis])
plt.scatter(x, y)
plt.plot(x, yfit, 'b')
plt.title('Lasso regression for price against grade (houses without renovation)')
plt.show()

x = house_data[house_data['yr_renovated'] != 0]['grade']
y = house_data[house_data['yr_renovated'] != 0]['price']
gauss_model = make_pipeline(GaussianFeatures(5), Lasso(alpha=0.1))
gauss_model.fit(x[:, np.newaxis], y)

yfit = gauss_model.predict(x[:, np.newaxis])
plt.scatter(x, y)
plt.plot(x, yfit, 'b')
plt.title('Lasso regression for price against grade (houses with renovation)')
plt.show()

# Ridge regression
x = house_data[house_data['yr_renovated'] == 0]['grade']
y = house_data[house_data['yr_renovated'] == 0]['price']
gauss_model = make_pipeline(GaussianFeatures(5), Ridge(alpha=0.1))
gauss_model.fit(x[:, np.newaxis], y)

yfit = gauss_model.predict(x[:, np.newaxis])
plt.scatter(x, y)
plt.plot(x, yfit, 'b')
plt.title('Ridge regression for price against grade (houses with renovation)')
plt.show()

x = house_data[house_data['yr_renovated'] == 0]['grade']
y = house_data[house_data['yr_renovated'] == 0]['price']
gauss_model = make_pipeline(GaussianFeatures(5), Ridge(alpha=0.1))
gauss_model.fit(x[:, np.newaxis], y)

yfit = gauss_model.predict(x[:, np.newaxis])
plt.scatter(x, y)
plt.plot(x, yfit, 'b')
plt.title('Ridge regression for price against grade (houses without renovation)')
plt.show()

