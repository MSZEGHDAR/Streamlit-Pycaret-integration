import streamlit as st
import pandas as pd
import numpy as np
import os
from ydata_profiling import ProfileReport
from pycaret.classification import setup, compare_models, predict_model, pull, save_model, load_model
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def save_file(file):
    with open(os.path.join(os.getcwd(), file.name), "wb") as f:
        f.write(file.getbuffer())
    return st.success(f"File {file.name} saved successfully.")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def main():
    st.set_page_config( page_title="Streamlit/Pycaret integration")
    
    st.sidebar.title("Navigate")
    
    choice = st.sidebar.radio("", ['Home', 'Upload', 'Profiling', 'ML'])
    
    if choice == 'Home':
        home_page()
    elif choice == 'Upload':
        upload_page()
    elif choice == 'Profiling':
        profiling_page()
    elif choice == 'ML':
        ml_page()

def load_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Failed to load image from {url}. Error: {str(e)}")
        return None

def home_page():
    st.title("Welcome")
    
    st.write("""
    This app allows you to:
    1. Upload and manage CSV files
    2. Generate profile reports for your data
    3. Train and compare machine learning models
    4. Make predictions using the best model""")

    st.subheader("How to use it")

    st.write("""
    1. Start by uploading your CSV file in the 'Upload' section.
    2. Use the 'Profiling' section to get insights about your data.
    3. In the 'ML' section, select your target variable and train models.
    4. Export the model, make predictions and download results.
    5. Have fun!
    """)
    
    logos = {
        "Streamlit": "https://avatars.githubusercontent.com/u/45109972?v=4",
        "Pandas Profiling": "https://avatars.githubusercontent.com/u/35166984?s=280&v=4",
        "Scikit-learn": "https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/scikit-learn/scikit-learn.png",
        "Plotly": "https://images.plot.ly/logo/new-branding/plotly-logomark.png",
        "PyCaret": "https://avatars.githubusercontent.com/u/58118658?v=4"
    }

    st.subheader("Libraries and tools used in this project:")

    cols = st.columns(len(logos))

    for col, (name, url) in zip(cols, logos.items()):
        with col:
            img = load_image(url)
            if img:
                st.image(img, width=100, caption=name)
            else:
                st.write(f"{name}")

def upload_page():
    st.title("Upload CSV")
    file = st.file_uploader("Choose a CSV file", type="csv")

    if file:
        save_file(file)

    saved_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv')]
    file_to_load = st.selectbox("Select a CSV file for operations", saved_files)
    
    col1, col2 = st.columns(2)
    
    if col1.button("Load Data"):
        try:
            df = load_data(file_to_load)
            st.session_state['df'] = df
            st.dataframe(df)
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    if col2.button("Delete Data"):
        try:
            os.remove(file_to_load)
            st.success(f"Deleted {file_to_load}")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error deleting file: {str(e)}")

def profiling_page():
    st.title('Profile Report')
    saved_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv')]
    file_to_load = st.selectbox("Select a CSV file for operations", saved_files)

    if st.button("Generate Profile Report"):
        try:
            with st.spinner("Generating report..."):
                df = pd.read_csv(file_to_load)
                pr = ProfileReport(df, explorative=True)
                st_profile_report = pr.to_html()
                st.components.v1.html(st_profile_report, height=600, scrolling=True)
                
            st.download_button(
                label="Download Profile Report",
                data=pr.to_html(),
                file_name="profile_report.html",
                mime="text/html"
            )
        except Exception as e:
            st.error(f"Error generating profile report: {str(e)}")

def ml_page():
    st.title("Machine Learning")
    if 'df' not in st.session_state:
        st.error("Please load data in the 'Upload' section first.")
    else:
        data = st.session_state['df']
        st.write("Sample of dataset:", data.head())

        use_subset = st.checkbox("Use a subset of data for faster processing")
        if use_subset:
            max_size = len(data)
            subset_size = st.slider("Select subset size", min_value=min(100, max_size), max_value=max_size, value=min(1000, max_size))
            if subset_size < max_size:
                data = data.sample(n=subset_size, random_state=42)
            else:
                st.warning(f"Using full dataset ({max_size} rows) as subset size equals or exceeds total rows.")

        target = st.selectbox("Select target variable", data.columns)
        
        class_distribution = Counter(data[target])
        st.write("Class distribution:", class_distribution)
        
        min_samples = min(class_distribution.values())
        if min_samples < 2:
            st.warning(f"The least populated class has only {min_samples} sample(s). This may cause issues with model training.")
            
            handle_imbalance = st.radio(
                "How would you like to handle this?",
                ("Remove rare classes", "Apply oversampling (SMOTE)", "Proceed anyway")
            )
            
            if handle_imbalance == "Remove rare classes":
                min_samples_threshold = st.slider("Minimum samples per class", min_value=2, max_value=10, value=2)
                data = data[data[target].isin([cls for cls, count in class_distribution.items() if count >= min_samples_threshold])]
                st.write("Updated class distribution:", Counter(data[target]))
            elif handle_imbalance == "Apply oversampling (SMOTE)":
                X = data.drop(columns=[target])
                y = data[target]
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                data = pd.concat([X_resampled, y_resampled], axis=1)
                st.write("Updated class distribution after SMOTE:", Counter(data[target]))
        
        if st.button('Setup Model'):
            try:
                with st.spinner('Setting up model...'):
                    s = setup(data=data, target=target, session_id=42)
                    st.session_state['setup'] = s
                st.success('Setup complete. Ready to train models.')
            except Exception as e:
                st.error(f"Error during setup: {str(e)}")
                st.error("Please try adjusting your data or selecting a different target variable.")
                
        if 'setup' in st.session_state:
            if st.button('Train and Compare Models'):
                try:
                    with st.spinner('Training models...'):
                        # 3 best models
                        best_models = compare_models(n_select=3)
                        st.session_state['best_models'] = best_models
                        model_results = pull()
                        st.dataframe(model_results)
                        
                        fig = px.bar(model_results, x=model_results.index, y='Accuracy')
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error during model comparison: {str(e)}")
            
            if 'best_models' in st.session_state:
                best_models = st.session_state['best_models']
                model_names = [str(model).split('(')[0] for model in best_models]
                selected_model = st.selectbox("Select model to save", model_names, index=0)
                
                if st.button('Save Selected Model'):
                    try:
                        model_index = model_names.index(selected_model)
                        save_model(best_models[model_index], f'best_model_{selected_model}')
                        st.success(f"{selected_model} saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                
                st.subheader("Make Predictions")
                prediction_option = st.radio("Choose prediction method", ["Use loaded data", "Input custom data"])
                
                if prediction_option == "Use loaded data":
                    if st.button('Make Predictions on Loaded Data'):
                        try:
                            model_index = model_names.index(selected_model)
                            predictions = predict_model(best_models[model_index], data=data)
                            st.write(f"Predictions from {selected_model}:")
                            st.dataframe(predictions)
                            st.download_button(
                                label=f"Download Predictions",
                                data=predictions.to_csv(index=False),
                                file_name=f"predictions_{selected_model}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")
                
                else:
                    st.write("Enter custom data for prediction:")
                    custom_data = {}
                    for column in data.columns:
                        if column != target:
                            custom_data[column] = st.text_input(f"Enter value for {column}")
                    
                    if st.button('Make Prediction on Custom Data'):
                        try:
                            custom_df = pd.DataFrame([custom_data])
                            model_index = model_names.index(selected_model)
                            prediction = predict_model(best_models[model_index], data=custom_df)
                            st.write(f"Prediction from {selected_model}:")
                            st.dataframe(prediction)
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()
