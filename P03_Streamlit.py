from datetime import datetime
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Function to preprocess appointment data consistently with training
def preprocess_appointments(appointments, encoder, scaler):
    # Impute missing categorical values
    categorical_cols = appointments.select_dtypes(include=['object']).columns
    imputer = SimpleImputer(strategy='most_frequent')
    appointments[categorical_cols] = imputer.fit_transform(appointments[categorical_cols])

    # Convert categorical columns to strings
    appointments[categorical_cols] = appointments[categorical_cols].astype(str)

    # Encoding categorical variables using the loaded encoder
    encoded_cols = encoder.transform(appointments[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate encoded columns and drop original categorical columns
    appointments = pd.concat([appointments, encoded_df], axis=1).drop(columns=categorical_cols)

    # Standardizing numerical features using the loaded scaler
    numerical_cols = appointments.select_dtypes(include=['int64', 'float64']).columns
    appointments[numerical_cols] = scaler.transform(appointments[numerical_cols])

    return appointments

# Function to load the entire pipeline (model and preprocessors)
def load_pipeline(pipeline_path="best_pipeline.pkl"):
    with open(pipeline_path, "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

def predict_no_show(pipeline, appointment_data, preserved_columns):
    try:
        # Ensure that MRN and APPT_ID are included in the prediction data
        appointment_data_with_preserved_columns = pd.concat([appointment_data, preserved_columns[['APPT_ID', 'MRN']]], axis=1)

        # Predict the no-show and probabilities directly using the pipeline
        predictions = pipeline.predict(appointment_data_with_preserved_columns)
        probabilities = pipeline.predict_proba(appointment_data_with_preserved_columns)[:, 1]

        # Add predictions and probabilities to the preserved columns
        preserved_columns['no_show_probability'] = probabilities
        preserved_columns['no_show_prediction'] = preserved_columns['no_show_probability'].apply(
            lambda x: 'Yes' if x >= 0.5 else 'No'
        )

        # Remove commas from MRN and APPT_ID
        preserved_columns['MRN'] = preserved_columns['MRN'].astype(str).str.replace(',', '')
        preserved_columns['APPT_ID'] = preserved_columns['APPT_ID'].astype(str).str.replace(',', '')

        # Remove patient_id column if it exists
        if 'patient_id' in preserved_columns.columns:
            preserved_columns = preserved_columns.drop(columns=['patient_id'])

        # Reorder columns to place MRN before APPT_ID
        column_order = ['MRN', 'APPT_ID', 'CLINIC', 'APPT_DATE', 'no_show_probability', 'no_show_prediction']
        preserved_columns = preserved_columns[column_order]

        return preserved_columns
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def retrieve_appointments(CLINIC, start_date, end_date, file_path="CHLA_clean_data_2024_Appointments.csv"):
    # Load the data
    appointments = pd.read_csv(file_path)
    
    # Strip any extra spaces in column names
    appointments.columns = appointments.columns.str.strip()
    
    # Ensure the necessary columns exist
    for col in ['APPT_ID', 'MRN', 'CLINIC', 'APPT_DATE']:
        if col not in appointments.columns:
            st.error(f"The required column '{col}' is missing from the data file.")
            return None, None

    # Convert the appointment date to datetime format
    appointments['APPT_DATE'] = pd.to_datetime(appointments['APPT_DATE'])
    
    # Filter data based on clinic name and date range
    filtered_appointments = appointments[
        (appointments['CLINIC'] == CLINIC) &
        (appointments['APPT_DATE'] >= start_date) &
        (appointments['APPT_DATE'] <= end_date)
    ]

    # If no data matches the filter, return an error
    if filtered_appointments.empty:
        st.error("No appointments found for the given clinic and date range.")
        return None, None

    # Preserve necessary columns for display
    preserved_columns = filtered_appointments[['APPT_ID', 'MRN', 'CLINIC', 'APPT_DATE']].copy()

    # Drop non-predictive columns for model input
    filtered_appointments = filtered_appointments.drop(columns=['APPT_ID', 'MRN'], errors='ignore')
    
    return filtered_appointments, preserved_columns

# Example usage
# Remove or comment out the following line to stop displaying records at the top of the app
# appointments = retrieve_appointments("VALENCIA CARE CENTER", "2024-01-01", "2024-01-31")
# if appointments is not None:
#     st.write(appointments)


# Function to load the model using pickle
def load_trained_model(model_path="best_random_forest.pkl"):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model

# Load the feature names
def load_feature_names(file_path="feature_names.pkl"):
    with open(file_path, "rb") as file:
        feature_names = pickle.load(file)
    return feature_names

# Preprocess the input data to align with training features
def preprocess_and_align(appointments, feature_names):
    # Preprocess the appointments data
    X = preprocess_appointments(appointments)

    # Align columns with the trained feature names
    missing_cols = set(feature_names) - set(X.columns)
    for col in missing_cols:
        X[col] = 0  # Add missing columns with 0

    # Reorder columns to match the training set
    X = X[feature_names]

    return X


# Streamlit App
def run_app():
    st.title("CHLA Clinic Appointment No-Show Prediction")

    # Predefined clinic options
    clinic_options = [
        "VALENCIA CARE CENTER",
        "ARCADIA CARE CENTER"
    ]

    # User inputs: Clinic name and Date range
    CLINIC = st.selectbox("Select Clinic Name", clinic_options)
    
    start_date = st.date_input("Start Date", datetime(2024, 1, 1))
    end_date = st.date_input("End Date", datetime(2024, 1, 31))
    
    # Submit button
    if st.button("Get Predictions"):
        # Convert date objects to datetime format
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Step 1: Retrieve the appointments
        appointments, preserved_columns = retrieve_appointments(CLINIC, start_date, end_date)

        if appointments is None or preserved_columns is None:
            st.error("Unable to retrieve appointments due to missing columns.")
            return

        # Load the trained pipeline
        pipeline = load_pipeline()

        # Step 2: Make predictions
        predictions = predict_no_show(pipeline, appointments, preserved_columns)

        # Display the results without the record indicator
        if predictions is not None:
            st.subheader("No-Show Predictions")
            st.dataframe(
                predictions.style.hide(axis="index")  # Hides the row index
            )

        # Optionally: Allow users to download the results as a CSV
        st.download_button(
            label="Download Predictions",
            data=predictions.to_csv(index=False),
            file_name="no_show_predictions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    run_app()
