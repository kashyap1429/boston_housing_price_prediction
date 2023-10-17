import pickle
import numpy as np
import streamlit as st

gradient_boosting_model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# st.set_page_config(
#     page_title="Boston House Price Prediction",
#     page_icon=" "
# )

st.title("Boston House Price Prediction 🏡")

CRIM = st.number_input('CRIM - per capita crime rate by town')
ZN = st.number_input('ZN - proportion of residential land zoned for lots over 25,000 sq.ft.')
INDUS = st.number_input('INDUS - proportion of non-retail business acres per town.')
CHAS = st.number_input('CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)')
NOX = st.number_input('NOX - nitric oxides concentration (parts per 10 million)')
RM = st.number_input('RM - average number of rooms per dwelling')
AGE = st.number_input('AGE - proportion of owner-occupied units built prior to 1940')
DIS = st.number_input('DIS - weighted distances to five Boston employment centres')
RAD = st.number_input('RAD - index of accessibility to radial highways')
TAX = st.number_input('TAX - full-value property-tax rate per $10,000')
PTRATIO = st.number_input('PTRATIO - pupil-teacher ratio by town')
B = st.number_input('B- 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')
LSTAT = st.number_input('LSTAT - % lower status of the population')

if st.button('Predict'):
    # Preprocess the input data
    input_data = np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = gradient_boosting_model.predict(input_data_scaled)
    
    # Display the prediction
    st.write(f'The predicted house price is: {prediction[0]}')

# Run the Streamlit app
if __name__ == '__main__':
    st.run()
