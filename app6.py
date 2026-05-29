#%%
import streamlit as st
import joblib
import pandas as pd

#%%

model = joblib.load('saved_model.pkl')

store_encoder = joblib.load('label_encoder_store.pkl')
category_encoder = joblib.load('label_encoder_category.pkl')

#%%

st.set_page_config(
    page_title='Coffee Shop ML App',
    layout='centered'
)
st.title('☕ Coffee Shop Sales Prediction')

st.write('Predict Total Bill using Machine Learning')

#%%

transaction_qty = st.number_input(
    'Transaction Quantity',
    min_value=1,
    max_value=20,
    value=2
)

unit_price = st.number_input(
    'Unit Price',
    min_value=1.0,
    max_value=20.0,
    value=5.0
)

store_location = st.selectbox(
    'Store Location',
    store_encoder.classes_
)

product_category = st.selectbox(
    'Product Category',
    category_encoder.classes_
)

hour = st.slider(
       'Hour',
    0,
    23,
    10
)

month = st.slider(
    'Month',
    1,
    12,
    6
)

weekday = st.slider(
    'Weekday',
    0,
    6,
    3
)

#%%

store_encoded = store_encoder.transform([store_location])[0]

category_encoded = category_encoder.transform([
    product_category
])[0]

#%%
if st.button('Predict Bill'):

    input_data = pd.DataFrame({
        'transaction_qty': [transaction_qty],
        'unit_price': [unit_price],
        'store_location_encoded': [store_encoded],
        'product_category_encoded': [category_encoded],
        'hour': [hour],
        'month': [month],
        'weekday': [weekday]
    })

    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Total Bill: ${prediction:.2f}')

#%%

st.write('Built by Sandeep Gupta 🚀')