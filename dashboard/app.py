import streamlit as st
import pandas as pd
import joblib
import io

st.title("Anomoly Detection Dashboard")

uploaded_file = st.file_uploader("Upload Network Log CSV")

if uploaded_file is not None:
    try:
        # Ensure the file is read properly as text
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        st.write("Data Preview:", df.head())

        model = joblib.load('../models/random_forest.pkl')

        # Drop 'label' column if it's in the uploaded file
# Run prediction
        X = df.select_dtypes(include='number').drop(columns=['label'], errors='ignore')
        predictions = model.predict(X)
        df['Anomaly'] = predictions

        # Show only anomaliesç
        anomalies = df[df["Anomaly"] == 1]

        st.subheader("Detected Anomalies")
        st.write(f"Found {len(anomalies)} anomalies out of {len(df)} total rows.")
        st.dataframe(anomalies)

        # Optional: Download anomalies
        csv = anomalies.to_csv(index=False).encode('utf-8')
        st.download_button("Download Anomalies as CSV", csv, "anomalies.csv", "text/csv")   

    except Exception as e:
        st.error(f"Error processing file: {e}")