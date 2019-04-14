import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

cars_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', header=None)

# Function to make a list to include all non-unknown continuous data
def transform(column_no):
    data = []
    for x in cars_data[cars_data.columns[column_no]]:
        try:
            x = float(x)
            data.append(x)
        except ValueError:
            continue
    return data

# Finding mean for continuous data with missing values
ave_normalized_loss = np.mean(transform(1))
ave_bore = np.mean(transform(18))
ave_stroke = np.mean(transform(19))
ave_horsepower = np.mean(transform(21))
ave_peak_rpm = np.mean(transform(22))
ave_price = np.mean(transform(25))


'''1. Replace all the missing values of continuous data (normalized-losses, bore, stroke, horsepower, peak-rpm, price)
    by the mean value of data for each variable.'''

cars_data[cars_data.columns[1]] = cars_data[cars_data.columns[1]].replace('?', ave_normalized_loss)
cars_data[cars_data.columns[18]] = cars_data[cars_data.columns[18]].replace('?', ave_bore)
cars_data[cars_data.columns[19]] = cars_data[cars_data.columns[19]].replace('?', ave_stroke)
cars_data[cars_data.columns[21]] = cars_data[cars_data.columns[21]].replace('?', ave_horsepower)
cars_data[cars_data.columns[22]] = cars_data[cars_data.columns[22]].replace('?', ave_peak_rpm)
cars_data[cars_data.columns[25]] = cars_data[cars_data.columns[25]].replace('?', ave_price)

most_freq_numOfDoors = cars_data[cars_data.columns[5]].value_counts().index.values[0]

cars_data[cars_data.columns[5]] = cars_data[cars_data.columns[5]].replace('?', most_freq_numOfDoors)

'''2. Convert categorical data into numerical data: Categorical data in column 1, column 6 and column 16 can be converted into numerical data'''

convert_numOfDoors = {'two': 2, 'four': 4}
convert_numOfCylinders = {'eight': 8, 'four': 4, 'five': 5, 'six': 6, 'three': 3, 'twelve': 12, 'two': 2}
cars_data[cars_data.columns[0]] = cars_data[cars_data.columns[0]].astype(int)
cars_data[cars_data.columns[5]].replace(convert_numOfDoors, inplace=True)
cars_data[cars_data.columns[15]].replace(convert_numOfCylinders, inplace=True)

# Print data after cleaning
print(cars_data)

'''3. Exploratory analysis on the initial data to find out outliers, data skew'''

# Initialize cars data without any missing values replaced
cars_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data', header=None)

# Convert numerical data from strings to floats
def convert_to_float(cars_raw, column_no):
    for column in column_no:
        cars_raw = cars_raw[cars_raw[cars_raw.columns[column]] != '?']
        cars_raw[cars_raw.columns[column]] = cars_raw[cars_raw.columns[column]].astype(float)
    return cars_raw
cars_raw_new = convert_to_float(cars_raw, [1, 18, 19, 21, 22, 25])

# Scatter plot and linear regression

plt.scatter(x=cars_raw_new[cars_raw_new.columns[13]], y=cars_raw_new[cars_raw_new.columns[21]])
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_raw_new[cars_raw_new.columns[13]], cars_raw_new[cars_raw_new.columns[21]])
plt.plot(cars_raw_new[cars_raw_new.columns[13]], intercept + slope*cars_raw_new[cars_raw_new.columns[13]], 'r')
plt.title('Correlation of horsepower against curb weight')
plt.xlabel('curb weight')
plt.ylabel('horsepower')
plt.show()
print('r-value:', r_value)

plt.scatter(x=cars_raw_new[cars_raw_new.columns[13]], y=cars_raw_new[cars_raw_new.columns[25]])
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_raw_new[cars_raw_new.columns[13]], cars_raw_new[cars_raw_new.columns[25]])
plt.plot(cars_raw_new[cars_raw_new.columns[13]], intercept + slope*cars_raw_new[cars_raw_new.columns[13]], 'r')
plt.title('Correlation of price against curb weight')
plt.xlabel('curb weight')
plt.ylabel('price')
plt.show()
print('r-value:', r_value)

plt.scatter(x=cars_raw_new[cars_raw_new.columns[13]], y=cars_raw_new[cars_raw_new.columns[22]])
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_raw_new[cars_raw_new.columns[13]], cars_raw_new[cars_raw_new.columns[22]])
plt.plot(cars_raw_new[cars_raw_new.columns[13]], intercept + slope*cars_raw_new[cars_raw_new.columns[13]], 'r')
plt.title('Correlation of peak-rpm against curb weight')
plt.xlabel('curb weight')
plt.ylabel('peak-rpm')
plt.show()
print('r-value:', r_value)

plt.scatter(x=cars_raw_new[cars_raw_new.columns[9]], y=cars_raw_new[cars_raw_new.columns[22]])
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_raw_new[cars_raw_new.columns[9]], cars_raw_new[cars_raw_new.columns[22]])
plt.plot(cars_raw_new[cars_raw_new.columns[9]], intercept + slope*cars_raw_new[cars_raw_new.columns[9]], 'r')
plt.title('Correlation of peak-rpm against wheel base')
plt.xlabel('wheel base')
plt.ylabel('peak-rpm')
plt.show()
print('r-value:', r_value)

plt.scatter(x=cars_raw_new[cars_raw_new.columns[9]], y=cars_raw_new[cars_raw_new.columns[21]])
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_raw_new[cars_raw_new.columns[9]], cars_raw_new[cars_raw_new.columns[21]])
plt.plot(cars_raw_new[cars_raw_new.columns[9]], intercept + slope*cars_raw_new[cars_raw_new.columns[9]], 'r')
plt.title('Correlation of housepower against wheel base')
plt.xlabel('wheel base')
plt.ylabel('housepower')
plt.show()
print('r-value:', r_value)

plt.scatter(x=cars_raw_new[cars_raw_new.columns[9]], y=cars_raw_new[cars_raw_new.columns[25]])
slope, intercept, r_value, p_value, std_err = stats.linregress(cars_raw_new[cars_raw_new.columns[9]], cars_raw_new[cars_raw_new.columns[25]])
plt.plot(cars_raw_new[cars_raw_new.columns[9]], intercept + slope*cars_raw_new[cars_raw_new.columns[9]], 'r')
plt.title('Correlation of price against wheel base')
plt.xlabel('wheel base')
plt.ylabel('price')
plt.show()
print('r-value:', r_value)

# Distribution plot

plt.hist(cars_raw_new[cars_raw_new.columns[13]], color = 'blue', edgecolor = 'black', bins=20)
plt.title('Distribution of curb weight')
plt.show()

plt.hist(cars_raw_new[cars_raw_new.columns[10]], color = 'orange', edgecolor = 'black', bins=20)
plt.title('Distribution of length')
plt.show()

plt.hist(cars_raw_new[cars_raw_new.columns[11]], color = 'purple', edgecolor = 'black', bins=20)
plt.title('Distribution of width')
plt.show()

plt.hist(cars_raw_new[cars_raw_new.columns[12]], color = 'yellow', edgecolor = 'black', bins=20)
plt.title('Distribution of height')
plt.show()

plt.hist(cars_raw_new[cars_raw_new.columns[9]], color = 'green', edgecolor = 'black', bins=20)
plt.title('Distribution of wheel base')
plt.show()

plt.hist(cars_raw_new[cars_raw_new.columns[0]], color = 'magenta', edgecolor = 'black', bins=6)
plt.title('Distribution of symboling')
plt.show()

plt.hist(cars_raw_new[cars_raw_new.columns[25]], color = 'red', edgecolor = 'black', bins=50)
plt.title('Distribution of car price')

#plt.show()

print(cars_raw_new.describe())

# Qualitative analysis
# Pick several factors to group by

by_body_style = cars_raw_new.groupby(cars_raw_new.columns[6])
by_fuel_type = cars_raw_new.groupby(cars_raw_new.columns[3])
by_engine_location = cars_raw_new.groupby(cars_raw_new.columns[14])
by_fuel_system = cars_raw_new.groupby(cars_raw_new.columns[17])
by_num_of_cylinders = cars_raw_new.groupby(cars_raw_new.columns[15])

body_style_mpg = by_body_style[cars_raw_new.columns[23], cars_raw_new.columns[24]].mean()
body_style_mpg.plot(kind='barh', title='Bar plots of body style against mean city-mpg and highway-mpg').legend(['city-mpg', 'highway-mpg'])
plt.show()

body_style_bore_stroke_ratio = by_body_style[cars_raw_new.columns[18], cars_raw_new.columns[19]].mean()
body_style_bore_stroke_ratio.plot(kind='barh', title='Bar plots of body style against bore and stroke').legend(['bore', 'stroke'])
plt.show()

body_style_bore_stroke_ratio = by_body_style[cars_raw_new.columns[18]].mean()/by_body_style[cars_raw_new.columns[19]].mean()
body_style_bore_stroke_ratio.plot(kind='barh', title='Bar plots of body style against mean bore/stroke ratio')
plt.show()

body_style_price = by_body_style[cars_raw_new.columns[25]].mean()
body_style_price.plot(kind='barh', title='Bar plots of body style against mean price')
plt.show()

fuel_type_mpg = by_fuel_type[cars_raw_new.columns[23], cars_raw_new.columns[24]].mean()
fuel_type_mpg.plot(kind='barh', title='Bar plots of fuel type against mean city-mpg and highway-mpg').legend(['city-mpg', 'highway-mpg'])
plt.show()

fuel_type_bore_stroke_ratio = by_fuel_type[cars_raw_new.columns[18], cars_raw_new.columns[19]].mean()
fuel_type_bore_stroke_ratio.plot(kind='barh', title='Bar plots of fuel type against bore and stroke').legend(['bore', 'stroke'])
plt.show()

fuel_type_bore_stroke_ratio = by_fuel_type[cars_raw_new.columns[18]].mean()/by_fuel_type[cars_raw_new.columns[19]].mean()
fuel_type_bore_stroke_ratio.plot(kind='barh', title='Bar plots of fuel type against mean bore/stroke ratio')
plt.show()

fuel_type_price = by_fuel_type[cars_raw_new.columns[25]].mean()
fuel_type_price.plot(kind='barh', title='Bar plots of fuel type against mean price')
plt.show()
                                 
engine_location_mpg = by_engine_location[cars_raw_new.columns[23], cars_raw_new.columns[24]].mean()
engine_location_mpg.plot(kind='barh', title='Bar plots of engine location against mean city-mpg and highway-mpg').legend(['city-mpg', 'highway-mpg'])
plt.show()

engine_location_bore_stroke = by_engine_location[cars_raw_new.columns[18], cars_raw_new.columns[19]].mean()
engine_location_bore_stroke.plot(kind='barh', title='Bar plots of engine location against bore and stroke').legend(['bore', 'stroke'])
plt.show()

engine_location_bore_stroke_ratio = by_engine_location[cars_raw_new.columns[18]].mean()/by_engine_location[cars_raw_new.columns[19]].mean()
engine_location_bore_stroke_ratio.plot(kind='barh', title='Bar plots of engine location against mean bore/stroke ratio')
plt.show()

engine_location_price = by_engine_location[cars_raw_new.columns[25]].mean()
engine_location_price.plot(kind='barh', title='Bar plots of engine location against mean price')
plt.show()

fuel_system_mpg = by_fuel_system[cars_raw_new.columns[23], cars_raw_new.columns[24]].mean()
fuel_system_mpg.plot(kind='barh', title='Bar plots of fuel system against mean city-mpg and highway-mpg').legend(['city-mpg', 'highway-mpg'])
plt.show()

fuel_system_bore_stroke = by_fuel_system[cars_raw_new.columns[18], cars_raw_new.columns[19]].mean()
fuel_system_bore_stroke.plot(kind='barh', title='Bar plots of fuel system against bore and stroke').legend(['bore', 'stroke'])
plt.show()

fuel_system_bore_stroke_ratio = by_fuel_system[cars_raw_new.columns[18]].mean()/by_fuel_system[cars_raw_new.columns[19]].mean()
fuel_system_bore_stroke_ratio.plot(kind='barh', title='Bar plots of fuel system against mean bore/stroke ratio')
plt.show()

fuel_system_price = by_fuel_system[cars_raw_new.columns[25]].mean()
fuel_system_price.plot(kind='barh', title='Bar plots of fuel system against mean price')
plt.show()

num_of_cylinders_mpg = by_num_of_cylinders[cars_raw_new.columns[23], cars_raw_new.columns[24]].mean()
num_of_cylinders_mpg.plot(kind='barh', title='Bar plots of num of cylinders against mean city-mpg and highway-mpg').legend(['city-mpg', 'highway-mpg'])
plt.show()

num_of_cylinders_bore_stroke = by_num_of_cylinders[cars_raw_new.columns[18], cars_raw_new.columns[19]].mean()
num_of_cylinders_bore_stroke.plot(kind='barh', title='Bar plots of num of cylinders against bore and stroke').legend(['bore', 'stroke'])
plt.show()

num_of_cylinders_bore_stroke_ratio = by_num_of_cylinders[cars_raw_new.columns[18]].mean()/by_num_of_cylinders[cars_raw_new.columns[19]].mean()
num_of_cylinders_bore_stroke_ratio.plot(kind='barh', title='Bar plots of num of cylinders against mean bore/stroke ratio')
plt.show()

num_of_cylinders_price = by_num_of_cylinders[cars_raw_new.columns[25]].mean()
num_of_cylinders_price.plot(kind='barh', title='Bar plots of num of cylinders against mean price')
plt.show()

cars_raw_new.boxplot(column=cars_raw_new.columns[13], by=cars_raw_new.columns[0])
plt.title('Box plot of curb weight by symboling')
plt.show()