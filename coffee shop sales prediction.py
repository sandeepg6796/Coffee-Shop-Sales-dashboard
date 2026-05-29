#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import joblib
# %%
df = pd.read_excel(r'C:\Users\sande\Desktop\Excel Coffee Sales Analysis\Raw Dataset.xlsx')

# %%
print(df.head())
#%%
df['total_bill'] = df['transaction_qty'] * df['unit_price']

#%%
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

#%%
df['month'] = df['transaction_date'].dt.month

df['day'] = df['transaction_date'].dt.day

df['weekday'] = df['transaction_date'].dt.weekday

#%%
df['transaction_time'] = pd.to_datetime(
    df['transaction_time'].astype(str)
)

#%%
df['hour'] = df['transaction_time'].dt.hour
#%%
print(df.isnull().sum())

#%%

numerical_columns = ['transaction_qty', 'unit_price']

for col in numerical_columns:
    df[col].fillna(df[col].median(), inplace=True)

#%%

categorical_columns = ['store_location', 'product_category']

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

#%%

store_encoder = LabelEncoder()
category_encoder = LabelEncoder()
#%%

df['store_location_encoded'] = store_encoder.fit_transform(
    df['store_location']
)
#%%
df['product_category_encoded'] = category_encoder.fit_transform(
    df['product_category']
)
#%%

joblib.dump(store_encoder, 'label_encoder_store.pkl')
joblib.dump(category_encoder, 'label_encoder_category.pkl')

#%%
X = df[[
    'transaction_qty',
    'unit_price',
    'store_location_encoded',
    'product_category_encoded',
    'hour',
    'month',
    'weekday'
]]

y = df['total_bill']

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print('Training Shape:', X_train.shape)
print('Testing Shape:', X_test.shape)

#%%

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

#%%

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

#%%

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

#%%

def evaluate_model(y_true, y_pred, model_name):

    mae = mean_absolute_error(y_true, y_pred)

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_true, y_pred)
    print('\n=================================')
    print(f'{model_name} Results')
    print('=================================')
    print('MAE :', mae)
    print('MSE :', mse)
    print('RMSE:', rmse)
    print('R2 Score:', r2)

#%%

evaluate_model(y_test, linear_predictions, 'Linear Regression')

#%%

evaluate_model(y_test, rf_predictions, 'Random Forest')

#%%

evaluate_model(y_test, xgb_predictions, 'XGBoost')

#%%

importance = rf_model.feature_importances_

feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print('\nFeature Importance')
print(importance_df)

#%%

plt.figure(figsize=(10, 5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title('Feature Importance')

plt.show()

#%%

joblib.dump(rf_model, 'saved_model.pkl')

print('\nModel Saved Successfully!')
# %%
