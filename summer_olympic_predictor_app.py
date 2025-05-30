import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("olympics_data.csv")  # Change this to your CSV if needed
    medal_data = df.groupby(['Year', 'Country'])['Medal'].count().reset_index()
    medal_data.rename(columns={'Medal': 'Medal_Count'}, inplace=True)
    le = LabelEncoder()
    medal_data['Country_Code'] = le.fit_transform(medal_data['Country'])
    return medal_data, le

data, le = load_data()

# Train model
X = data[['Year', 'Country_Code']]
y = data['Medal_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏅 Olympic Medal Predictor")
st.write("Predict the number of medals a country might win in a given Olympic year.")

country_list = list(le.classes_)
country = st.selectbox("Select Country", country_list)
year = st.slider("Select Year", min_value=1980, max_value=2032, step=4, value=2020)

if st.checkbox("📊 Show medal trend for selected country"):
    country_df = data[data['Country'] == country]
    st.line_chart(country_df.set_index('Year')['Medal_Count'])

# Prediction section
if st.button("🔮 Predict Future Medals"):
    try:
        country_code = le.transform([country])[0]
        prediction_df = pd.DataFrame({'Year': [year], 'Country_Code': [country_code]})
        predicted_medals = model.predict(prediction_df)[0]
        st.success(f"🥇 Predicted medals for {country} in {year}: {round(predicted_medals)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
