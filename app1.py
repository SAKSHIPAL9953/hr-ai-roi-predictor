import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration & Styling
st.set_page_config(page_title="HR AI Adoption ROI Predictor", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size:32px; font-weight:bold; color: #1E3A8A; text-align: center; margin-bottom: 20px; }
    .section-title { font-size:20px; font-weight:bold; color: #1E3A8A; margin-top: 15px; margin-bottom: 10px; }
    .output-box { background-color: #F3F4F6; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# STEP 1: BACKGROUND ENGINE (Model Loading)
# ----------------------------------------------------
@st.cache_resource
def load_models():
    # Load the serialized trained model and feature columns
    model = joblib.load("ai_productivity_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns

try:
    model, model_columns = load_models()
except Exception as e:
    st.error("Error loading model files. Please ensure '.pkl' files are in the same directory.")
    st.stop()

# ----------------------------------------------------
# STEP 2: USER INTERFACE DESIGN
# ----------------------------------------------------
st.markdown("<div class='main-title'>HR AI Adoption ROI & Productivity Predictor</div>", unsafe_allow_html=True)
st.write("Enter the new employee metrics below to instantly forecast their Productivity Gain and Financial ROI.")

# Create a layout with two main columns (Left for inputs, Right for outputs)
col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader("📊 Employee & AI Work Profile Input")
    
    # Create 3 sub-columns inside the input form for a cleaner look
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=35)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with f_col2:
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=6500)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=30000, value=14000)
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=70)
        percent_hike = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=15)
        overtime = st.selectbox("Overtime", ["No", "Yes"])

    with f_col3:
        ai_usage = st.slider("AI Usage Level (1=Low, 3=High)", 1, 3, 3)
        uses_ai = st.selectbox("Uses AI Tools", ["Yes", "No"])
        ai_training = st.selectbox("AI Training Level", ["50+ Hours", "21-50 Hours", "1-20 Hours", "0 Hours"])
        ai_sentiment = st.selectbox("AI Adoption Sentiment", ["Supportive", "Neutral", "Resistant"])
        perf_rating = st.slider("Performance Rating", 3, 4, 3)
        work_life = st.slider("Work Life Balance (1=Bad, 4=Best)", 1, 4, 3)

    # Miscellaneous variables mapped directly to avoid single-row dummy variance
    distance = 8 if department == "Research & Development" else 12
    education = 3
    env_sat = 3
    job_inv = 3
    job_lvl = 2
    job_sat = 3
    num_comp = 2
    rel_sat = 3
    stock_lvl = 1
    total_years = 10
    training_last_year = 3
    years_at_company = 5
    years_curr_role = 3
    years_promotion = 1
    years_manager = 3
    attrition = "No"
    travel = "Travel_Rarely"

with col2:
    st.subheader("🎯 Live Prediction & ROI")
    st.write("Click below to calculate real-time insights.")
    
    if st.button("🚀 Run Working Model", type="primary"):
        
        # Map categorical training strings to numeric representations used in training
        training_map = {'0 Hours': 0, '1-20 Hours': 10, '21-50 Hours': 35, '50+ Hours': 60}
        ai_training_numeric = training_map[ai_training]
        
        # Prepare input dict
        input_data = {
            'Age': age, 'Attrition': attrition, 'BusinessTravel': travel, 'DailyRate': daily_rate,
            'Department': department, 'DistanceFromHome': distance, 'Education': education,
            'EducationField': 'Life Sciences', 'EnvironmentSatisfaction': env_sat, 'Gender': gender,
            'HourlyRate': hourly_rate, 'JobInvolvement': job_inv, 'JobLevel': job_lvl, 'JobRole': job_role,
            'JobSatisfaction': job_sat, 'MaritalStatus': marital_status, 'MonthlyIncome': monthly_income,
            'MonthlyRate': monthly_rate, 'NumCompaniesWorked': num_comp, 'OverTime': overtime,
            'PercentSalaryHike': percent_hike, 'PerformanceRating': perf_rating,
            'RelationshipSatisfaction': rel_sat, 'StockOptionLevel': stock_lvl, 'TotalWorkingYears': total_years,
            'TrainingTimesLastYear': training_last_year, 'WorkLifeBalance': work_life,
            'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_curr_role,
            'YearsSinceLastPromotion': years_promotion, 'YearsWithCurrManager': years_manager,
            'AI_Usage_Level': ai_usage, 'Uses_AI_Tools': uses_ai, 'AI_Training_Hours': ai_training_numeric,
            'AI_Adoption_Sentiment': ai_sentiment
        }
        
        # Robust Alignment: Create full dummy structure matching the model's exact signature
        input_df = pd.DataFrame([input_data])
        
        # Build completely structured empty array based on model signature columns
        encoded_template = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Map specific values manually into the template columns
        for col in model_columns:
            if col in input_df.columns:
                encoded_template[col] = input_df[col].values[0]
            else:
                # Handle categorical text alignment for one-hot encoding columns
                for cat_feat in ['Department', 'JobRole', 'MaritalStatus', 'Gender', 'OverTime', 'Uses_AI_Tools', 'AI_Adoption_Sentiment', 'Attrition', 'BusinessTravel']:
                    if col.startswith(cat_feat + "_"):
                        val = str(input_data[cat_feat])
                        if col == f"{cat_feat}_{val}":
                            encoded_template[col] = 1

        # Predict using aligned data signature
        pred_code = model.predict(encoded_template)[0]
        
        # Map Code back to labels and mid-values for Business Logic calculations
        label_map = {0: ("0%", 0), 3: ("1-5%", 3), 8: ("6-10%", 8), 12: ("10%+", 12)}
        gain_label, gain_val = label_map.get(pred_code, ("0%", 0))
        
        # ----------------------------------------------------
        # CORE FEATURE ENGINEERING BUSINESS LOGIC (Live Run)
        # ----------------------------------------------------
        # Calculate time saved per week (assuming standard 40-hour baseline)
        estimated_time_saved = (gain_val / 100) * 40
        
        # Calculate standard hourly cost based on monthly income / 160 hours
        hourly_cost = monthly_income / 160
        
        # Calculate overall monetary value of time saved for the organization
        monetary_benefit = estimated_time_saved * hourly_cost
        
        # Setting a generic base baseline cost configuration for simulated AI tools
        simulated_ai_cost = 45 
        
        # Calculate core ROI percentage metrics
        roi = ((monetary_benefit - simulated_ai_cost) / simulated_ai_cost) * 100
        
        # Render the calculated output dashboard container blocks
        st.markdown("<div class='output-box'>", unsafe_allow_html=True)
        st.metric(label="📊 Predicted Productivity Gain", value=gain_label)
        st.metric(label="⏱️ Estimated Weekly Time Saved", value=f"{estimated_time_saved:.2f} Hours")
        st.metric(label="💰 Weekly Monetary Benefit to Company", value=f"${monetary_benefit:.2f}")
        
        if roi > 0:
            st.metric(label="📈 Estimated AI ROI", value=f"{roi:.2f}%", delta="Positive Return")
        else:
            st.metric(label="📉 Estimated AI ROI", value=f"{roi:.2f}%", delta="Negative/Breakeven", delta_color="inverse")
        st.markdown("</div>", unsafe_allow_html=True)