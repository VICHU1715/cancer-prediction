import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import os

# Attempt to read the dataset with error handling
def load_data():
    try:
        # Attempt to read the CSV file
        data = pd.read_csv('dataset.csv', engine='python')
        return data
    except MemoryError:
        st.error("MemoryError: The file might be too large to fit into memory.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"ParserError: {str(e)}")
        return None

data = load_data()

# Check if data is loaded successfully
if data is not None:
    # Preprocessing functions
    def preprocess_data():
        data.fillna(method='ffill', inplace=True)

        # Encode categorical variables
        label_encoders = {}
        for column in ['Gender', 'Symptoms', 'Medical_History', 'Ethnicity', 'Lymph_Node_Involvement', 'Cancer_Type', 'Generation_Report']:
            if column in data.columns:
                label_encoder = LabelEncoder()
                data[column] = label_encoder.fit_transform(data[column].astype(str))
                label_encoders[column] = label_encoder

        # Feature scaling
        scaler = StandardScaler()
        X = data.drop(['Patient_ID', 'Cancer_Type'], axis=1, errors='ignore')  # 'errors' param added to ignore missing columns
        X_scaled = scaler.fit_transform(X)
        return X, X_scaled, label_encoders, scaler

    # Model training for cancer type classification
    def train_cancer_type_model():
        X, X_scaled, label_encoders, scaler = preprocess_data()
        y = data['Cancer_Type']
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_scaled, y)
        feature_names = X.columns
        return rf_model, label_encoders, scaler, feature_names

    # Function to determine the cancer stage based on tumor size
    def determine_stage(tumor_size_cm):
        if 1 <= tumor_size_cm < 2:
            return "Stage 1"
        elif 2 <= tumor_size_cm < 3:
            return "Stage 2"
        elif 3 <= tumor_size_cm < 4:
            return "Stage 3"
        elif 4 <= tumor_size_cm <= 5:
            return "Stage 4"
        else:
            return "Stage Unknown"

    # Function to recommend medication based on cancer type and stage
    def recommend_medication(cancer_type, stage):
        medications = {
            "Melanoma": {"Stage 1": "Immunotherapy", "Stage 2": "Targeted Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Palliative Care"},
            "Colorectal": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
            "Prostate": {"Stage 1": "Surgery", "Stage 2": "Radiation Therapy", "Stage 3": "Hormone Therapy", "Stage 4": "Chemotherapy"},
            "Breast": {"Stage 1": "Surgery", "Stage 2": "Radiation Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Targeted Therapy"},
            "Ovarian": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
            "Lymphoma": {"Stage 1": "Chemotherapy", "Stage 2": "Radiation Therapy", "Stage 3": "Immunotherapy", "Stage 4": "Palliative Care"},
            "Pancreatic": {"Stage 1": "Surgery", "Stage 2": "Chemotherapy", "Stage 3": "Targeted Therapy", "Stage 4": "Palliative Care"},
            "Lung": {"Stage 1": "Surgery", "Stage 2": "Radiation Therapy", "Stage 3": "Chemotherapy", "Stage 4": "Targeted Therapy"},
        }
        return medications.get(cancer_type, {}).get(stage, "No specific medication found")

    # Function to predict life expectancy based on cancer type and stage
    def predict_life_expectancy(cancer_type, stage, with_medication=True):
        life_expectancy_with_medication = {
            "Melanoma": {"Stage 1": 10, "Stage 2": 8, "Stage 3": 5, "Stage 4": 2},
            "Colorectal": {"Stage 1": 12, "Stage 2": 10, "Stage 3": 7, "Stage 4": 3},
            "Prostate": {"Stage 1": 14, "Stage 2": 11, "Stage 3": 6, "Stage 4": 2},
            "Breast": {"Stage 1": 15, "Stage 2": 12, "Stage 3": 8, "Stage 4": 4},
            "Ovarian": {"Stage 1": 13, "Stage 2": 10, "Stage 3": 6, "Stage 4": 3},
            "Lymphoma": {"Stage 1": 16, "Stage 2": 13, "Stage 3": 8, "Stage 4": 3},
            "Pancreatic": {"Stage 1": 11, "Stage 2": 8, "Stage 3": 5, "Stage 4": 2},
            "Lung": {"Stage 1": 9, "Stage 2": 7, "Stage 3": 4, "Stage 4": 1},
        }

        life_expectancy_without_medication = {
            "Melanoma": {"Stage 1": 7, "Stage 2": 5, "Stage 3": 3, "Stage 4": 1},
            "Colorectal": {"Stage 1": 9, "Stage 2": 7, "Stage 3": 4, "Stage 4": 1},
            "Prostate": {"Stage 1": 10, "Stage 2": 8, "Stage 3": 4, "Stage 4": 1},
            "Breast": {"Stage 1": 11, "Stage 2": 9, "Stage 3": 5, "Stage 4": 2},
            "Ovarian": {"Stage 1": 9, "Stage 2": 7, "Stage 3": 4, "Stage 4": 2},
            "Lymphoma": {"Stage 1": 13, "Stage 2": 10, "Stage 3": 5, "Stage 4": 2},
            "Pancreatic": {"Stage 1": 8, "Stage 2": 6, "Stage 3": 3, "Stage 4": 1},
            "Lung": {"Stage 1": 6, "Stage 2": 4, "Stage 3": 2, "Stage 4": 0.5},
        }

        if with_medication:
            return life_expectancy_with_medication.get(cancer_type, {}).get(stage, "Data not available")
        else:
            return life_expectancy_without_medication.get(cancer_type, {}).get(stage, "Data not available")

    # Function to plot life expectancy with and without medication
    def plot_life_expectancy(cancer_type, stage):
        with_medication = predict_life_expectancy(cancer_type, stage, with_medication=True)
        without_medication = predict_life_expectancy(cancer_type, stage, with_medication=False)
        
        labels = ['With Medication', 'Without Medication']
        values = [with_medication, without_medication]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(labels, values, color=['green', 'red'])
        ax.set_ylabel('Life Expectancy (Years)')
        ax.set_title(f'Life Expectancy for {cancer_type} - {stage}')
        st.pyplot(fig)

    # Function to plot average survival years by cancer type
    def plot_survival_by_cancer_type():
        cancer_types = ["Melanoma", "Colorectal", "Prostate", "Breast", "Ovarian", "Lymphoma", "Pancreatic", "Lung"]
        stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        survival_years = {cancer: [predict_life_expectancy(cancer, stage) for stage in stages] for cancer in cancer_types}

        fig, ax = plt.subplots(figsize=(10, 6))
        for cancer_type in cancer_types:
            ax.plot(stages, survival_years[cancer_type], marker='o', label=cancer_type)

        ax.set_xlabel('Stage')
        ax.set_ylabel('Average Survival Years')
        ax.set_title('Average Survival Years by Cancer Type and Stage')
        ax.legend()
        st.pyplot(fig)

    # Function to plot cancer type increase over the years
    def plot_cancer_type_increase():
        years = list(range(2015, 2021))
        cancer_incidents = [np.random.randint(200, 500) for _ in years]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(years, cancer_incidents, marker='o', color='purple')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Cancer Cases')
        ax.set_title('Cancer Type Increase Over the Years')
        st.pyplot(fig)

    # Function to get the image path based on cancer type and stage
    def get_image_path(cancer_type, stage):
        base_dir = "path/to/your/images"  # Set this to the directory where images are stored
        image_file = f"{cancer_type}_{stage}.jpg"  # Assuming your images are named like "Melanoma_Stage_1.jpg"
        return os.path.join(base_dir, image_file)

    # Function to display the cancer image
    def display_cancer_image(cancer_type, stage):
        image_path = get_image_path(cancer_type, stage)
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=f"{cancer_type} - {stage}", use_column_width=True)
        else:
            st.warning(f"No image available for {cancer_type} - {stage}")

    # Streamlit app layout
    st.title("Cancer Prediction and Medication Recommendation")
    st.header("Cancer Type Classification")

    # Input fields for user data
    age = st.number_input("Age", min_value=0, max_value=100, value=50)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", options=["Caucasian", "African American", "Asian", "Hispanic"])
    tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=10.0, value=2.0)
    lymph_node_involvement = st.selectbox("Lymph Node Involvement", options=["Yes", "No"])
    symptoms = st.multiselect("Symptoms", options=["Fatigue", "Weight Loss", "Pain", "Fever", "Cough", "Night Sweats"])
    medical_history = st.multiselect("Medical History", options=["Diabetes", "Hypertension", "Heart Disease", "None"])

    # Cancer Type Prediction button
    if st.button("Predict Cancer Type"):
        try:
            rf_model, label_encoders, scaler, feature_names = train_cancer_type_model()

            # Collect user input into a DataFrame
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Ethnicity': [ethnicity],
                'Tumor_Size_cm': [tumor_size],
                'Lymph_Node_Involvement': [lymph_node_involvement],
                'Symptoms': [", ".join(symptoms)],
                'Medical_History': [", ".join(medical_history)]
            })

            # Encode input data
            for column, encoder in label_encoders.items():
                if column in input_data.columns:
                    input_data[column] = encoder.transform(input_data[column].astype(str))

            # Feature scaling
            input_scaled = scaler.transform(input_data)

            # Predict cancer type
            cancer_type_pred = rf_model.predict(input_scaled)[0]
            predicted_cancer_type = label_encoders['Cancer_Type'].inverse_transform([cancer_type_pred])[0]

            # Determine cancer stage
            stage = determine_stage(tumor_size)

            # Recommend medication
            recommend_medication(predicted_cancer_type, stage)

            # Display cancer type prediction
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Cancer Type:** {predicted_cancer_type}")
            st.markdown(f"**Cancer Stage:** {stage}")

            # Display cancer image
            display_cancer_image(predicted_cancer_type, stage)

            # Predict and display life expectancy
            life_expectancy = predict_life_expectancy(predicted_cancer_type, stage)
            st.markdown(f"**Predicted Life Expectancy:** {life_expectancy} years")

            # Plot life expectancy
            plot_life_expectancy(predicted_cancer_type, stage)

            # Optionally, plot additional visualizations
            if st.checkbox("Show Additional Visualizations"):
                st.subheader("Additional Visualizations")
                plot_survival_by_cancer_type()
                plot_cancer_type_increase()

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

else:
    st.write("Awaiting input data for prediction.")

