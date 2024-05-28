import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load training data
X_train = np.load('/data/rf_X_train.npy')
y_train = np.load('/data/y_train.npy')

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# joblib.dump(rf_model, '../models/rf_model.pkl')
model_path = 'C:/Users/musab/Desktop/project/models/rf_model.pkl'

# Save the model
joblib.dump(rf_model, model_path)
# LightGBM
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
model_path = 'C:/Users/musab/Desktop/project/models/lgbm_model.pkl'

# joblib.dump(lgb_model, '../models/lgbm_model.pkl')
joblib.dump(lgb_model, model_path)

# XGBoost
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
xgb_model = xgb.XGBClassifier(
    colsample_bytree=0.8, gamma=0, learning_rate=0.01, max_depth=5,
    min_child_weight=1, n_estimators=500, objective='multi:softmax',
    subsample=0.8
)
xgb_model.fit(X_train, y_train_encoded)
model_path = 'C:/Users/musab/Desktop/project/models/xgb_model.pkl'

# joblib.dump(xgb_model, '../models/xgb_model.pkl')
joblib.dump(xgb_model, model_path)

# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # Load preprocessed data
# X_train = np.load('/data/X_train.npy')
# X_test = np.load('/data/X_test.npy')
# y_train = np.load('/data/y_train.npy')
# y_test = np.load('/data/y_test.npy')

# # Train model
# random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
# random_forest.fit(X_train, y_train)

# # Save model
# # joblib.dump(random_forest, '/models/model.pkl')
# model_path = 'C:/Users/musab/Desktop/project/models/model.pkl'

# # Save the model
# joblib.dump(random_forest, model_path)