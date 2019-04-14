#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
toy_data = pd.read_csv('toy_dataset.csv')

#toy_data.fillna(0)


#standard categorization of data
#print(pd.get_dummies(toy_data['Gender']))

minAge = toy_data['Age'].min()
meanAge = toy_data['Age'].mean()
stdAge = toy_data['Age'].std()
toy_data['z-score_age'] = (toy_data['Age'] - meanAge)/stdAge
print(toy_data.head())


# In[26]:


airquality = pd.read_csv('airquality.csv')

#replace a nan value with mean of ozone
#print(airquality['Ozone'].replace(np.nan, np.mean(airquality['Ozone'])))


most_freq_Ozone = airquality['Ozone'].value_counts().index.values[0]

airquality['Ozone'].fillna(most_freq_Ozone)



# In[ ]:





# In[ ]:




