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

CRIM = st.number_input('CRIM')
ZN = st.number_input('ZN')
INDUS = st.number_input('INDUS')
CHAS = st.number_input('CHAS')
NOX = st.number_input('NOX')
RM = st.number_input('RM')
AGE = st.number_input('AGE')
DIS = st.number_input('DIS')
RAD = st.number_input('RAD')
TAX = st.number_input('TAX')
PTRATIO = st.number_input('PTRATIO')
B = st.number_input('B')
LSTAT = st.number_input('LSTAT')

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
