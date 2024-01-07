from operator import index
import streamlit as st

from pycaret.regression import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 
st.markdown("<style>body { font-family: Arial, sans-serif; }</style>", unsafe_allow_html=True)
import matplotlib
matplotlib.use('Agg')
if os.path.exists('./sourcedata.csv'): 
    df = pd.read_csv('sourcedata.csv', index_col=None).astype(int)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoNickML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df)
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
       
        setup(df, target=chosen_target)
        setup_df = pull()
        print("setup_df_kartik",setup_df)
        setup_df['Value'] = pd.to_numeric(setup_df['Value'], errors='coerce')
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")