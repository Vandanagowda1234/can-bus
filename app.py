import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🚗 CAN Bus Intrusion Detection System")

# ✅ Cache data loading + training
@st.cache_data
def load_data():
    data = pd.read_csv("data/dataset.csv", header=None)

    data.columns = [
        "Timestamp", "ID", "DLC",
        "Data0", "Data1", "Data2", "Data3",
        "Data4", "Data5", "Data6", "Data7",
        "Label"
    ]

    data = data.dropna()

    # Convert hex to int
    data['ID'] = data['ID'].apply(lambda x: int(str(x), 16))

    for col in ['Data0','Data1','Data2','Data3','Data4','Data5','Data6','Data7']:
        data[col] = data[col].apply(lambda x: int(str(x), 16))

    # Take small sample
    data = data.sample(n=5000, random_state=42)

    return data


@st.cache_resource
def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X = X.select_dtypes(include=['number'])

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X.columns


# Load + train once
data = load_data()
model, columns = train_model(data)

# UI
st.subheader("Enter CAN Message Data")

input_data = []

for col in columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

st.write("")

# Predict
if st.button("Detect Attack"):
    prediction = model.predict([input_data])

    if prediction[0] == 0:
        st.success("✅ Normal Message")
    else:
        st.error("⚠️ Attack Detected")