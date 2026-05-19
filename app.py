import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Patch keras to handle quantization_config
def patch_keras():
    try:
        from keras.src.layers import layer
        original_init = layer.Layer.__init__
        
        def patched_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            original_init(self, *args, **kwargs)
        
        layer.Layer.__init__ = patched_init
    except:
        pass

patch_keras()

# Set page config
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .high-risk {
        color: #ff4444;
        font-weight: bold;
    }
    .low-risk {
        color: #44aa44;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and preprocessing files
@st.cache_resource
def load_models():
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    model = None
    
    # Try loading model.pkl first
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
            st.info("✓ Loaded model from model.pkl")
        except Exception as e:
            st.warning(f"Could not load model.pkl: {e}")
    
    # Fallback to model.h5 if pkl not available
    if model is None and os.path.exists('model.h5'):
        try:
            import tensorflow as tf
            from tensorflow.python.keras.utils import generic_utils
            
            # Try with minimal options
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = tf.keras.models.load_model('model.h5', compile=False, safe_mode=False)
            st.info("✓ Loaded model from model.h5")
        except Exception as e:
            try:
                # Last attempt with a completely different approach
                import h5py
                import tensorflow as tf
                
                # Load just the weights if config fails
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                model.load_weights('model.h5')
                st.info("✓ Loaded model weights from model.h5")
            except Exception as e2:
                st.error(f"Could not load any model: {str(e2)}")
                raise
    
    if model is None:
        raise FileNotFoundError("Neither model.pkl nor model.h5 found!")
    
    with open('one_hot_encoder_geo.pkl', 'rb') as file:
        label_encoder_geo = pickle.load(file)
    
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_geo, label_encoder_gender, scaler

# Load models
model, label_encoder_geo, label_encoder_gender, scaler = load_models()

# Main title
st.markdown("<h1 class='main-header'> Bank Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Add description
st.markdown("""
This application predicts whether a bank customer is likely to churn (leave the bank) based on their profile information.
Enter the customer details below and click the predict button to get the prediction.
""")

st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Customer Information")
    
    # Numerical inputs
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=10
    )
    
    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=45,
        step=1
    )
    
    tenure = st.slider(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5,
        step=1
    )
    
    balance = st.number_input(
        "Balance ($)",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )

with col2:
    st.subheader("Additional Details")
    
    # Categorical inputs
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        index=0
    )
    
    geography = st.selectbox(
        "Geography",
        options=["France", "Germany", "Spain"],
        index=0
    )
    
    num_of_products = st.slider(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )
    
    has_credit_card = st.selectbox(
        "Has Credit Card?",
        options=["Yes", "No"],
        index=0
    )

# Third column for more inputs
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    is_active_member = st.selectbox(
        "Is Active Member?",
        options=["Yes", "No"],
        index=1
    )
    
    estimated_salary = st.number_input(
        "Estimated Salary ($)",
        min_value=10000.0,
        value=75000.0,
        step=5000.0
    )

with col4:
    st.info(" **Tip:** Adjust the sliders and dropdowns to see how different factors affect churn prediction.")

st.markdown("---")

# Prediction button
if st.button(" Predict Churn", use_container_width=True):
    # Prepare the data
    try:
        # Create input dataframe
        input_data = {
            "CreditScore": credit_score,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": 1 if has_credit_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary,
            "Geography": geography
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Encode Gender
        input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])
        
        # One-hot encode Geography
        geo_encoded = label_encoder_geo.transform(input_df[["Geography"]])
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=label_encoder_geo.get_feature_names_out(["Geography"])
        )
        
        # Drop Geography and concatenate
        input_df = input_df.drop("Geography", axis=1)
        input_df = pd.concat(
            [input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
            axis=1
        )
        
        # Remove duplicate columns if any
        input_df = input_df.loc[:, ~input_df.columns.duplicated()]
        
        # Reindex to match scaler's expected features
        input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
        
        # Scale the features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction (handles both sklearn and TensorFlow models)
        try:
            # Try as TensorFlow model first
            prediction = model.predict(input_scaled, verbose=0)
            if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
                churn_probability = float(prediction[0][0])
            else:
                churn_probability = float(prediction[0])
        except (AttributeError, TypeError):
            # Fall back to sklearn model
            prediction = model.predict_proba(input_scaled)
            churn_probability = float(prediction[0][1])
        
        # Display results
        st.markdown("---")
        st.subheader(" Prediction Result")
        
        # Create result container
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if churn_probability > 0.5:
                st.error(f" **HIGH CHURN RISK**")
                st.markdown(f"### <span class='high-risk'>This customer is likely to churn</span>", unsafe_allow_html=True)
            else:
                st.success(f" **LOW CHURN RISK**")
                st.markdown(f"### <span class='low-risk'>This customer is likely to stay</span>", unsafe_allow_html=True)
        
        with result_col2:
            # Display probability as a gauge
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability*100:.2f}%",
                delta=None
            )
        
        # Show probability bar
        st.markdown("#### Churn Probability Distribution")
        try:
            prob_val = float(churn_probability)
            prob_val = max(0.01, min(0.99, prob_val))  # Clamp between 0.01 and 0.99 for column sizing
            col_prob1, col_prob2 = st.columns([prob_val, 1-prob_val])
            
            with col_prob1:
                st.markdown(f"<div style='background-color: #ff4444; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;'>{churn_probability*100:.2f}%</div>", unsafe_allow_html=True)
            
            with col_prob2:
                st.markdown(f"<div style='background-color: #44aa44; padding: 10px; border-radius: 5px; text-align: center; color: white; font-weight: bold;'>{(1-churn_probability)*100:.2f}%</div>", unsafe_allow_html=True)
        except:
            st.info(f"Churn Probability: {churn_probability*100:.2f}%")
        
        # Show input summary
        st.markdown("#### Customer Profile Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.write(f"**Credit Score:** {credit_score}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Age:** {age} years")
        
        with summary_col2:
            st.write(f"**Tenure:** {tenure} years")
            st.write(f"**Balance:** ${balance:,.2f}")
            st.write(f"**Products:** {num_of_products}")
        
        with summary_col3:
            st.write(f"**Credit Card:** {has_credit_card}")
            st.write(f"**Active Member:** {is_active_member}")
            st.write(f"**Salary:** ${estimated_salary:,.2f}")
    
    except Exception as e:
        st.error(f" Error making prediction: {str(e)}")
        st.write("Please check that all input files (model.h5, pickle files) are available.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p> Bank Churn Prediction System | Deep Learning Model | Powered by TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)


