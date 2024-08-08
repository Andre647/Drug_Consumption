
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from pages._1_Visualization import df
from pages._2_Conclusion import target_columns, hot_encode, scaler, df_v, metrics,metrics_completed

st.markdown("# Test Yourself")

age_value = st.selectbox("Age", sorted(df['Age'].unique()))
edu_value = st.selectbox("Education(remember: LS = Left School at): ",sorted(df['Education'].unique()))

Nscore = st.slider("Nscore", df['Nscore'].min(), df['Nscore'].max(), 0.)
Escore = st.slider("Escore", df['Escore'].min(), df['Escore'].max(), 0.)
Oscore = st.slider("Oscore", df['Oscore'].min(), df['Oscore'].max(), 0.)
Ascore = st.slider("Ascore", df['Ascore'].min(), df['Ascore'].max(), 0.)
Cscore = st.slider("Cscore", df['Cscore'].min(), df['Cscore'].max(), 0.)
impulsive = st.slider("Impulsive", df['Impulsive'].min(), df['Impulsive'].max(), 0.)
SS = st.slider("SS", df['SS'].min(), df['SS'].max(), 0.)

new_input = {
    'Age':[edu_value],
    'Education': [age_value],
    'Nscore': [Nscore],
    'Escore': [Escore],
    'Oscore': [Oscore],
    'Ascore': [Ascore],
    'Cscore': [Cscore],
    'Impulsive': [impulsive],
    'SS': [SS]
}
new_df = pd.DataFrame(new_input)
st.dataframe(new_df)
new_df_hot = pd.DataFrame(hot_encode.transform(new_df.select_dtypes(exclude='number')),columns=hot_encode.get_feature_names_out())
new_df = new_df_hot.join(new_df.select_dtypes('number'))
new_df_scaled = scaler.transform(new_df)
new_df_scaled = pd.DataFrame(new_df_scaled, columns=df_v.columns)
new_df = new_df_scaled

drug = st.selectbox("Select the Drug", sorted(target_columns))

drug_b = st.button("Reset")
if st.button("Predict", type="primary"):
    prediction = metrics_completed[drug]['model'].predict(new_df)
    if prediction == 1: st.write(f"**You might be a potencial *{drug}* user** :skull:")
    else: st.write(f"**The chances of you become a *{drug}* user its low** :star:")

