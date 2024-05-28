import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_multiclass_target(y):
    return np.where((y[:, 0] == 0) & (y[:, 1] == 0), 0,
                    np.where((y[:, 0] == 1) & (y[:, 1] == 1), 1,
                             np.where((y[:, 0] == 0) & (y[:, 1] == 1), 2, 3)))

# df = pd.read_csv('../data/dataset.csv')
df = pd.read_csv(r'C:\Users\musab\Desktop\project\data\ht_traning_1_csv.csv')  # Using raw string
df.dropna(inplace=True)
y = df.iloc[:, :2].values
X = df.iloc[:, 2:].values

y_multiclass = create_multiclass_target(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

data_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

np.save(os.path.join(data_dir, 'rf_X_train.npy'), X_train)
np.save(os.path.join(data_dir, 'xgb_X_train.npy'), X_train)
np.save(os.path.join(data_dir, 'lgbm_X_train.npy'), X_train)

np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

# np.save('/data/rf_X_train.npy', X_train)
# np.save('/data/xgb_X_train.npy', X_train)  # Assuming the same X_train for all models
# np.save('/data/lgbm_X_train.npy', X_train)

# np.save('/data/y_train.npy', y_train)
# np.save('/data/X_test.npy', X_test)
# np.save('/data/y_test.npy', y_test)

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Load data
# # df = pd.read_csv("hyperthyroidism3_FinalDataSet_April.csv")
# # df = pd.read_csv('C:\Users\musab\Desktop\project\data')
# # df = pd.read_csv('C:\\Users\\musab\\Desktop\\project\\data') 


# # Define X and y
# y = df.iloc[:, :2].values
# X = df.iloc[:, 2:].values

# def create_multiclass_target(y):
#     return np.where((y[:, 0] == 0) & (y[:, 1] == 0), 0,
#                     np.where((y[:, 0] == 1) & (y[:, 1] == 1), 1,
#                              np.where((y[:, 0] == 0) & (y[:, 1] == 1), 2, 3)))

# y_multiclass = create_multiclass_target(y)
# print("ihhi")
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)
# import os
# print(os.getcwd())
# # Save preprocessed data
# print("ihshi")
# # Save preprocessed data
# data_dir = os.path.join(os.getcwd(), 'data')
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)

# np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
# np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
# np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
# np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
