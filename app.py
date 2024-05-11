import streamlit as st

st.title('MPG ML Project')

#'displacement','horsepower','weight','acceleration'

displacement= st.number_input("Enter a value for Displacement",value=300,placeholder="Enter a value for Displacement")
horsepower= st.number_input("Enter a value for Horsepower",value=130,placeholder="Enter a value for Horsepower")
weight= st.number_input("Enter a value for Weight",value=3000,placeholder="Enter a value for Weight")
acceleration= st.number_input("Enter a value for Acceleration",value=12,placeholder="Enter a value for Acceleration")


import pickle

load_model = pickle.load(open('mpg_regression.sav','rb'))
prediction= load_model.predict([[displacement,horsepower,weight,acceleration]])
st.subheader(f'Prediction mpg value for above parameter is {prediction[0]}')
#st.write(prediction)
