import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration & Styling
st.set_page_config(page_title="HR AI Adoption ROI Predictor", layout="wide")

# Custom Professional Premium CSS
st.markdown("""
    <style>
    .main-title { font-size:36px; font-weight:bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size:16px; text-align: center; color: #555555; margin-bottom: 30px; }
    .card { background-color: #F8FAFC; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #E2E8F0; }
    .metric-container { background-color: #FFFFFF; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
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
# STEP 2: USER INTERFACE DESIGN (Premium Multi-Tab)
# ----------------------------------------------------
st.markdown("<div class='main-title'>💼 HR AI Adoption ROI Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Enterprise-grade Workforce Productivity Forecasting & Financial ROI Dashboard</div>", unsafe_allow_html=True)

# Define Tabs for a clean multi-page flow inside a single app
tab1, tab2, tab3 = st.tabs(["🧑‍💼 1. Employee Profile", "🤖 2. AI & Analytics Inputs", "🎯 3. Live ROI Dashboard"])

with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Personal & Professional Demographics")
    st.write("Fill in the standard HR profile details for the employee below:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", min_value=18, max_value=60, value=35)
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with col_b:
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=6500)
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=30000, value=14000)
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=70)
        percent_hike = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=15)
        overtime = st.selectbox("Overtime Working", ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.info("ℹ️ Next Step: Please click on the '2. AI & Analytics Inputs' tab at the top to configure AI details.")

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Toolkit Configuration & Sentiment")
    st.write("Specify how the employee interacts with corporate artificial intelligence software:")
    
    col_c, col_d = st.columns(2)
    with col_c:
        uses_ai = st.selectbox("Currently Uses AI Tools", ["Yes", "No"])
        ai_usage = st.slider("AI Integration Level (1=Low Integration, 3=High Workflow Dependency)", 1, 3, 3)
        
    with col_d:
        ai_training = st.selectbox("AI Training Received", ["50+ Hours", "21-50 Hours", "1-20 Hours", "0 Hours"])
        ai_sentiment = st.selectbox("Employee AI Adoption Sentiment", ["Supportive", "Neutral", "Resistant"])

    st.subheader("Performance & Culture Baseline")
    col_e, col_f = st.columns(2)
    with col_e:
        perf_rating = st.slider("Current Performance Rating", 3, 4, 3)
    with col_f:
        work_life = st.slider("Work Life Balance Score (1=Poor, 4=Excellent)", 1, 4, 3)
    st.markdown("</div>", unsafe_allow_html=True)

    # Miscellaneous structural mappings to prevent encoding variance
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

with tab3:
    st.subheader("Execute Predictive Analytics Model")
    st.write("Click the action button below to pass inputs into the Random Forest Engine and dynamically calculate ROI metrics.")
    
    # Large primary functional button
    run_prediction = st.button("🚀 Generate Live Prediction & Financial Impact Reports", type="primary", use_container_width=True)
    
    if run_prediction:
        # Map categorical training strings to numeric representations used in training
        training_map = {'0 Hours': 0, '1-20 Hours': 10, '21-50 Hours': 35, '50+ Hours': 60}
        ai_training_numeric = training_map[ai_training]
        
        # Prepare data dict
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
        
        # Robust Alignment: Create dummy matrix matching model sign columns
        input_df = pd.DataFrame([input_data])
        encoded_template = pd.DataFrame(0, index=[0], columns=model_columns)
        
        for col in model_columns:
            if col in input_df.columns:
                encoded_template[col] = input_df[col].values[0]
            else:
                for cat_feat in ['Department', 'JobRole', 'MaritalStatus', 'Gender', 'OverTime', 'Uses_AI_Tools', 'AI_Adoption_Sentiment', 'Attrition', 'BusinessTravel']:
                    if col.startswith(cat_feat + "_"):
                        val = str(input_data[cat_feat])
                        if col == f"{cat_feat}_{val}":
                            encoded_template[col] = 1

        # Predict target class code
        pred_code = model.predict(encoded_template)[0]
        
        # Map labels
        label_map = {0: ("0%", 0), 3: ("1-5%", 3), 8: ("6-10%", 8), 12: ("10%+", 12)}
        gain_label, gain_val = label_map.get(pred_code, ("0%", 0))
        
        # Business Intelligence Formulas (Feature Engineering)
        estimated_time_saved = (gain_val / 100) * 40
        hourly_cost = monthly_income / 160
        monetary_benefit = estimated_time_saved * hourly_cost
        simulated_ai_cost = 45 
        roi = ((monetary_benefit - simulated_ai_cost) / simulated_ai_cost) * 100
        
        # Render clean corporate metric layout dashboards
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Forecasted Optimization Insights")
        
        out_col1, out_col2 = st.columns(2)
        
        with out_col1:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(label="Predicted Productivity Gain", value=gain_label)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(label="Estimated Weekly Time Saved", value=f"{estimated_time_saved:.2f} Hours")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with out_col2:
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            st.metric(label="Weekly Monetary Benefit to Organization", value=f"${monetary_benefit:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
            if roi > 0:
                st.metric(label="Estimated AI Strategy ROI", value=f"{roi:.2f}%", delta="Positive Financial Return")
            else:
                st.metric(label="Estimated AI Strategy ROI", value=f"{roi:.2f}%", delta="Negative/Breakeven Cost", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Waiting for Input Execution. Please configure Tab 1 and Tab 2, then click the button above.")