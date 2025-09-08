import joblib
import streamlit as st
import pandas as pd

def start_app(model):
    st.title("House Price Prediction")
    size = float(st.number_input("Size (in square meters)", key="size"))
    nb_rooms = int(st.number_input("Number of rooms", key="nb_rooms", step=1))
    garden = int(st.checkbox("Has a Garden", key="garden"))

    if st.button("Predict Price"):
        input_data = pd.DataFrame([[size, nb_rooms, garden]], columns=['size', 'nb_rooms', 'garden'])
        prediction = model.predict(input_data)
        st.write(f"Predicted Price: ${prediction[0]:,.2f}")

def main():
    model = joblib.load("../model/regression.joblib")
    start_app(model)

main()
