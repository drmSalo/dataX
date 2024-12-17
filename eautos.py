import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
import numpy as np
import os
import locale
import altair as alt

# Read data
df = pd.read_csv("./EAutos.csv")
locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')

# Filter the data
countries = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", 
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", 
    "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", 
    "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
]

filtered_df = df[df['region'].isin(countries)]
filtered_df = filtered_df[filtered_df["unit"] == "Vehicles"]
filtered_df = filtered_df[filtered_df["parameter"] == "EV sales"]
filtered_df = filtered_df[filtered_df["year"] > 2016]
powertrain = ["PHEV", "BEV"]
filtered_df = filtered_df[filtered_df["powertrain"].isin(powertrain)]

# Linear regression model
model = LinearRegression()

# Plot function to predict EV adoption for a specific country
def predict_country(country_df, powertrain_type, pred_year):
    pwrt = powertrain_type
    country_df = country_df[country_df["powertrain"] == pwrt]
    x = country_df["year"].values.reshape(-1, 1)
    y = country_df["value"].values
    model.fit(x, y)
    y_pred = model.predict(x)

    
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    
    predicted_value = model.predict(np.array([[pred_year]]))
    return predicted_value[0]



# Streamlit app layout
st.title("EV Adoption Prediction")
st.write(f"Choose a country to see EV adoption predictions.")

year = st.select_slider("Select which year you want to predict", options=[2025, 2026, 2027, 2028, 2029, 2030])



country_name = st.selectbox("Select a Country", options=filtered_df['region'].unique())
image = f"./img/{country_name}.png"
if  os.path.exists(image):
    st.image(image, width=250)
else: 
    st.write(f"No image found for {country_name}")
    


ev_type = st.selectbox("Select a Type", options=filtered_df["powertrain"].unique())


country_df = filtered_df[filtered_df['region'] == country_name]
powertrain_df = country_df[country_df['powertrain'] == ev_type]




if country_df.empty:
    st.write(f"No data available for {country_name}.")
else:
    
    prediction = predict_country(country_df, ev_type, year)
    st.write(f"The predicted EV adoption value for {country_name} in {year} is {locale.format_string("%.0f", prediction, grouping=True)}")

chart = alt.Chart(powertrain_df).mark_line().encode(
    x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("value", title="Value")
).properties(title = "EV Sales Over Time", 
             width=600,  
             height=400 )


bars = alt.Chart(country_df).mark_bar().encode(
    x = alt.X("value", title="Quantity"),
    y = alt.Y("powertrain", title="Ev Type"),
    color = alt.condition(
    alt.datum.powertrain == "PHEV", 
    alt.value("green"),
    alt.value("lightblue")
    )
     ).properties(title="Powertrain Comparison", width=600, height=200)

st.altair_chart(chart)
st.altair_chart(bars)