import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
print(x.shape)
#plt.scatter(x, y)
#plt.show()

model = LinearRegression(fit_intercept=True)
# x has to be reshaped in place
#model.fit(x.reshape(-1,1), y)

model.fit(x[:, np.newaxis], y)

print(model.intercept_)
print(model.coef_)

newX = np.linspace(0,10,20)
newX.shape
newY = model.predict(newX[:, np.newaxis])
#plt.scatter(newX, newY)
#plt.show()

X = np.array([3,4,5])
poly = PolynomialFeatures(3, include_bias=False)
Y = np.dot(poly.fit_transform(X[:, None]), [1,2,3])
print(Y)
model.fit(poly.fit_transform(X[:, None]), Y)
print(model.intercept_)
print(model.coef_)

'''poly_model = make_pipeline(PolynomialFeatures(3), LinearRegression())
poly_model.fit(poly.fit_transform(X[:, None]), Y)'''


