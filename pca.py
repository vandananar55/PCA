# import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from columndescription import Description





z = Description()

data = pd.read_csv("/home/anmol/Downloads/bot_accounts.csv")
print(data)

# EXPLORE DATASET.......................

#print(data.shape)
print(data.columns)
# print(data.info())

data = data.drop(["NAME","EMAIL_ID","REGISTRATION_IPV4"],axis=1)
# print(data)

new_data=data.dropna(subset=['ISBOT'])
# print(new_data)

# print(new_data.columns)
# print(new_data.info())
# print(new_data.shape)


all_null = new_data .isnull().sum()
# print (all_null)

pd.set_option('mode.chained_assignment',None)

# handling null values................

categorical = [col for col in new_data.columns if new_data[col].dtypes == 'O']
# print("categorical colums are.....",categorical)

numerical = [col for col in new_data.columns if new_data[col].dtypes != 'O']
# print("numerical colums are.....",numerical)

for cat in categorical:
    # print(cat)
    z.category_columns(new_data,cat)


# print("for loop ended =====>>>")

for num in numerical:
    # print(num)
    z.numerical_columns(new_data,num)

# # WORKING ON GENDER


new_data.GENDER = [1 if i == "Male" else 0 for i in new_data.GENDER]
# print(new_data.GENDER)

# # # WORKING ON IS_GLOGIN

new_data.IS_GLOGIN = [1 if i == True else 0 for i in new_data.IS_GLOGIN]
# print(new_data.IS_GLOGIN)


# # # WORKING ON REGISTRATION_LOCATION
all_names = new_data['REGISTRATION_LOCATION'].unique()
# print("ALL NAMES =>>",all_names)
dic = dict((value,index) for index,value in enumerate(all_names))
# print("dict from all name =>",dic)

new_data['REGISTRATION_LOCATION'] = new_data['REGISTRATION_LOCATION'].map(dic)
# print(new_data['REGISTRATION_LOCATION'].tail(1000))
# print(new_data['REGISTRATION_LOCATION'].head(10000))

# # # print(new_data.info())

# # working on ISBOT..................
new_data.ISBOT = [1 if i == True else 0 for i in new_data.ISBOT]
# # print((new_data.ISBOT).unique())
# # print((new_data.ISBOT))



scaler = StandardScaler()
print(scaler.fit(new_data))

scaled_data = scaler.transform(new_data)
print(scaled_data)
print(scaled_data.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

data_pca = pca.fit_transform(scaled_data)
print(data_pca)
print(data_pca.shape)

print(pca.explained_variance_)