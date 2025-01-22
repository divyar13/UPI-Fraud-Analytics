import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('UPI_Fraud_Detection_updated.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the feature names used during training
with open('feature_columns.pkl', 'rb') as file:
    training_columns = pickle.load(file)

# Streamlit app
st.title("UPI Fraud Detection System")

# Single transaction inputs
st.header("Check a Single Transaction")
amount = st.number_input("Transaction Amount", min_value=0.0, max_value=50000.0, step=0.1)
transaction_type = st.selectbox("Transaction Type", ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"])
payment_gateway = st.selectbox("Payment Gateway", ["Bank of Data", "CReditPAY", "Dummy Bank", "Gamma Bank", "Other", "SamplePay", "Sigma Bank", "UPI Pay"])
transaction_state = st.selectbox("Transaction State", ["Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"])

# Preprocess input
if st.button("Check Transaction"):
    try:
        # Create a DataFrame for input
        input_data = pd.DataFrame(
            [[amount, transaction_type, payment_gateway, transaction_state]],
            columns=["Amount", "Transaction_Type", "Payment_Gateway", "Transaction_State"]
        )

        # One-hot encode the input data
        input_data = pd.get_dummies(input_data)

        # Align the input data with training columns
        input_data = input_data.reindex(columns=training_columns, fill_value=0)

        # Predict using the model
        prediction = model.predict(input_data)

        # Display the result
        if prediction[0] == 1:
            st.error("Fraudulent Transaction Detected!")
        else:
            st.success("Transaction is Safe.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Batch prediction from CSV
st.header("OR Upload a CSV File for Batch Prediction")
csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
if st.button("Check CSV"):
    if csv_file is not None:
        try:
            # Load the CSV file
            uploaded_data = pd.read_csv(csv_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(uploaded_data)

            # Ensure the required columns exist
            required_columns = ["Amount", "Transaction_Type", "Payment_Gateway", "Transaction_State"]
            if not all(column in uploaded_data.columns for column in required_columns):
                st.error("Uploaded CSV does not have the required columns.")
            else:
                # One-hot encode the input data
                uploaded_data = pd.get_dummies(uploaded_data)

                # Align the input data with training columns
                uploaded_data = uploaded_data.reindex(columns=training_columns, fill_value=0)

                # Make predictions
                predictions = model.predict(uploaded_data)

                # Add predictions to the DataFrame
                uploaded_data["Prediction"] = ["Fraud" if p == 1 else "Safe" for p in predictions]
                st.write("Prediction Results:")
                st.dataframe(uploaded_data)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.error("Please upload a CSV file.")
