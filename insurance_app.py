import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import xgboost as xgb

# Load the model
model = pickle.load(open('xgb_reg_best.pkl', 'rb'))

st.title("Medical Insurance Cost Prediction")

tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 Visualizations", "📖 About"])
with tab1:
    st.write("This is a web application that predicts the medical insurance cost based on user inputs.")
    st.sidebar.header("Input Features")

    def user_input_features():
     age = st.sidebar.slider('Age', 18, 100, 30)
     bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
     children = st.sidebar.slider('Number of Children', 0, 5, 1)
     sex = st.sidebar.selectbox('Sex',('male', 'female'))
     smoker = st.sidebar.selectbox('Smoker', ('yes', 'no'))
     region = st.sidebar.selectbox('Region', ('northeast', 'northwest', 'southeast', 'southwest'))

     data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male':1 if sex == 'male' else 0,
        'smoker_yes': 1 if smoker == 'yes' else 0,
        'region_northeast': 1 if region == 'northeast' else 0,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southwest': 1 if region == 'southwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
     }

     features = pd.DataFrame(data, index=[0])
     return features

input_df = user_input_features()
st.subheader("User Input")
st.write(input_df)

model_columns = model.get_booster().feature_names
input_df = input_df[model_columns]

prediction = model.predict(input_df)
st.subheader("Predicted Insurance Cost")
st.write(f"${prediction[0]:.2f}")

with tab2:
    st.header("Data Visualizations")

    with st.expander("Feature Importance"):
     fig1, ax1 = plt.subplots(figsize=(10, 6))
     plot_importance(model, ax=ax1)
     st.pyplot(fig1)

with tab3:
    st.header("About This App")
    st.write("""
    - Built using Streamlit
    - Machine Learning model: XGBoost Regressor
    - Predicts Medical Insurance Costs
    - Provides various Data Visualizations for better insights
    """)
    st.info("Developed with ❤️ by Ramya")