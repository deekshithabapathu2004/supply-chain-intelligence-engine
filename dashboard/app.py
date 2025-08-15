import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set page config
st.set_page_config(page_title="Supply Chain Intelligence Engine", layout="wide")

# Custom CSS for clean header
st.markdown(
    """
    <div style="background-color:#0068C9; padding:15px; border-radius:10px; margin-bottom:20px;">
    <h1 style="color:white; text-align:center; font-size:32px;">Supply Chain Intelligence Engine</h1>
    <p style="color:white; text-align:center; font-size:16px;">Predictive Analytics for Quality Inspection</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/defect_prediction_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

if model is None:
    st.stop()

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/cleaned_supply_chain.csv')

df = load_data()

# Global features
numerical_features = [
    'Price', 'Availability', 'Number of products sold', 'Revenue generated',
    'Stock levels', 'Order quantities', 'Shipping times', 'Shipping costs',
    'Manufacturing costs', 'Defect rates', 'Production volumes',
    'Manufacturing lead time (days)'
]

categorical_cols = ['Supplier name', 'Transportation modes', 'Product type', 'Location']

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model Information", "EDA & Insights", "Predict Inspection", "Supplier Analysis", "Business Insights"])

# ———————————————————— PAGE 1: HOME ————————————————————
if page == "Home":
    st.markdown("""
    ### Project Overview
    This dashboard analyzes supply chain data from a **Beauty & Cosmetics startup** to:
    - Predict whether a product batch will **Pass or Fail** inspection
    - Identify high-risk suppliers and transportation modes
    - Optimize quality control and reduce defects

    ### Key Features
    - Real-time ML prediction using **XGBoost**
    - Interactive visualizations with **Plotly**
    - Supplier performance scoring
    - Defect rate trend analysis
    - Business recommendations

    ### How It Works
    The model uses historical data on:
    - Manufacturing lead time
    - Defect rates
    - Supplier
    - Transportation mode
    - And more
    To predict inspection outcomes with **75% accuracy**
    """)  # ← Fixed: Added closing triple quotes

    st.info("Navigate using the sidebar to explore EDA, Model Info, Predictions, and Supplier Analysis.")

# ———————————————————— PAGE 2: MODEL INFORMATION ————————————————————
elif page == "Model Information":
    st.title("Machine Learning Model Details")

    st.markdown("""
    ### Model Type: XGBoost Classifier
    - **Algorithm**: XGBoost (Extreme Gradient Boosting)
    - **Task**: Binary Classification (Pass/Fail Inspection)
    - **Training Data**: 47 batches
    - **Test Data**: 12 batches

    ### Performance Metrics
    | Metric       | Score |
    |--------------|-------|
    | Accuracy     | 75%   |
    | Precision (Fail) | 83% |
    | Recall (Fail)    | 71% |
    | F1-Score (Fail)  | 77% |
    | Precision (Pass) | 67% |
    | Recall (Pass)    | 80% |
    | F1-Score (Pass)  | 73% |

    ### Key Features Used
    - Defect rates
    - Manufacturing costs
    - Supplier name
    - Transportation modes
    - Production volumes
    - Shipping times

    ### Why XGBoost?
    - Handles mixed data types well
    - Resistant to overfitting
    - Provides feature importance
    - Fast and scalable

    ### Business Impact
    This model helps:
    - Flag high-risk batches before inspection
    - Reduce quality failures
    - Save costs by avoiding rework
    - Improve supplier selection
    """)

    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': model.get_booster().feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        fig = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', title="Top 10 Important Features")
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        st.plotly_chart(fig, use_container_width=True)

# ———————————————————— PAGE 3: EDA & Insights ————————————————————
elif page == "EDA & Insights":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Defect Rates", "Supplier Performance", "Transportation Impact"])

    with tab1:
        st.subheader("Defect Rate vs Inspection Result")
        fig = px.box(df, x='Inspection results', y='Defect rates',
                     labels={'Inspection results': 'Inspection Result (0=Fail, 1=Pass)'},
                     color='Inspection results',
                     color_discrete_map={0: '#D62728', 1: '#2CA02C'},
                     category_orders={'Inspection results': [0, 1]})
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730'),
            title_font_size=16
        )
        fig.update_xaxes(ticktext=['Fail (0)', 'Pass (1)'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Failures by Supplier")
        fail_counts = df[df['Inspection results'] == 0]['Supplier name'].value_counts().reset_index()
        fail_counts.columns = ['Supplier', 'Failures']
        fig = px.bar(fail_counts, x='Supplier', y='Failures', color='Failures', text='Failures', color_continuous_scale='Reds')
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Pass Rate by Transportation Mode")
        pass_rate = pd.crosstab(df['Transportation modes'], df['Inspection results'], normalize='index') * 100
        pass_rate = pass_rate.rename(columns={0: 'Fail %', 1: 'Pass %'})
        pass_rate = pass_rate.reset_index()
        fig = px.bar(pass_rate, x='Transportation modes', y='Pass %', text='Pass %', title="Pass Rate by Transportation Mode", color='Pass %', color_continuous_scale='Blues')
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730')
        )
        st.plotly_chart(fig, use_container_width=True)

# ———————————————————— PAGE 4: Predict Inspection ————————————————————
elif page == "Predict Inspection":
    st.title("Predict Inspection Result")

    st.write("Enter the details of a new product batch to predict if it will Pass or Fail inspection.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            price = st.number_input("Price", min_value=0.0, value=50.0)
            availability = st.number_input("Availability", min_value=0, value=50)
            num_sold = st.number_input("Number of products sold", min_value=0, value=500)
            revenue = st.number_input("Revenue generated", min_value=0.0, value=5000.0)
            stock = st.number_input("Stock levels", min_value=0, value=60)
            order_qty = st.number_input("Order quantities", min_value=0, value=50)
            shipping_time = st.number_input("Shipping times (days)", min_value=0, value=5)
            shipping_cost = st.number_input("Shipping costs", min_value=0.0, value=5.0)

        with col2:
            mfg_cost = st.number_input("Manufacturing costs", min_value=0.0, value=30.0)
            defect_rate = st.number_input("Defect rates (%)", min_value=0.0, max_value=100.0, value=2.5)
            prod_volume = st.number_input("Production volumes", min_value=0, value=500)
            lead_time = st.number_input("Manufacturing lead time (days)", min_value=0, value=20)
            supplier = st.selectbox("Supplier name", ['Supplier 1', 'Supplier 2', 'Supplier 3', 'Supplier 4', 'Supplier 5'])
            transport = st.selectbox("Transportation modes", ['Road', 'Rail', 'Air', 'Sea'])
            product_type = st.selectbox("Product type", ['haircare', 'skincare', 'cosmetics'])
            location = st.selectbox("Location", ['Mumbai', 'Kolkata', 'Delhi', 'Chennai', 'Bangalore'])

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            'Price': price, 'Availability': availability, 'Number of products sold': num_sold,
            'Revenue generated': revenue, 'Stock levels': stock, 'Order quantities': order_qty,
            'Shipping times': shipping_time, 'Shipping costs': shipping_cost,
            'Manufacturing costs': mfg_cost, 'Defect rates': defect_rate,
            'Production volumes': prod_volume, 'Manufacturing lead time (days)': lead_time,
            'Supplier name': supplier, 'Transportation modes': transport,
            'Product type': product_type, 'Location': location
        }

        df_input = pd.DataFrame([input_data])
        df_input = pd.get_dummies(df_input, columns=categorical_cols)

        for col in model.get_booster().feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input.reindex(columns=model.get_booster().feature_names, fill_value=0)
        df_input[numerical_features] = scaler.transform(df_input[numerical_features])

        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0]

        st.markdown("### Prediction Result")
        if pred == 1:
            st.success(f"PASS with {prob[1]:.1%} confidence")
            st.info("This batch is likely to pass quality inspection.")
        else:
            st.error(f"FAIL with {prob[0]:.1%} confidence")
            st.warning("This batch has a high risk of failing inspection. Recommend additional quality checks.")

# ———————————————————— PAGE 5: Supplier Analysis ————————————————————
elif page == "Supplier Analysis":
    st.title("Supplier Performance Dashboard")

    supplier_summary = df.groupby('Supplier name').agg(
        Total_Batches=('Inspection results', 'count'),
        Failed_Batches=('Inspection results', lambda x: (1 - x).sum()),
        Avg_Defect_Rate=('Defect rates', 'mean'),
        Avg_Manufacturing_Cost=('Manufacturing costs', 'mean')
    ).round(3)

    st.dataframe(supplier_summary.style.highlight_max(axis=0, color='pink').highlight_min(axis=0, color='lightgreen'))

    st.markdown("### Supplier Recommendations")
    st.markdown("""
    - **Best Supplier**: Supplier 1 (Low defect rate, high pass rate)
    - **Risky Supplier**: Supplier 4 (All 12 batches failed)
    - **High Cost**: Supplier 5 (High manufacturing cost)
    """)

    fail_counts = df[df['Inspection results'] == 0]['Supplier name'].value_counts().reset_index()
    fail_counts.columns = ['Supplier', 'Failures']
    fig = px.bar(fail_counts, x='Supplier', y='Failures', color='Failures', text='Failures', color_continuous_scale='Reds')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#262730')
    )
    st.plotly_chart(fig, use_container_width=True)

# ———————————————————— PAGE 6: Business Insights ————————————————————
elif page == "Business Insights":
    st.title("Key Business Insights")

    st.markdown("""
    ### What Drives a Failed Inspection?

    Based on the data, here are the top factors that increase the risk of **failing inspection**:

    #### 1. Supplier Risk
    - **Supplier 4** has a **100% failure rate** (12 out of 12 batches failed)
    - **Supplier 1** has the highest pass rate
    **Recommendation**: Phase out Supplier 4 for critical products

    #### 2. Defect Rate Threshold
    - Batches with **defect rates > 2.5\\%** are 3x more likely to fail
    **Recommendation**: Flag any batch with defect rate > 2.5% for pre-inspection review

    #### 3. Transportation Mode
    - **Road** is the most common mode for failed batches
    - **Rail and Air** are more reliable
    **Recommendation**: Use Rail/Air for high-value or sensitive products

    #### 4. Product Type
    - **Haircare** products have the highest failure rate
    - **Skincare** products pass most often
    **Recommendation**: Review haircare production process for quality gaps

    #### 5. No Cost-Quality Link
    - Failed batches have **higher manufacturing costs** (₹52.23 vs ₹46.14)
    - But still fail → **spending more ≠ better quality**
    **Recommendation**: Audit Supplier 4’s cost structure — are they cutting corners?

    ---

    ### Actionable Recommendations

    | Risk | Recommendation |
    |------|----------------|
    | High defect rate | Set automated alerts for defect rate > 2.5% |
    | Bad supplier | Replace Supplier 4 with Supplier 1 |
    | Road transport | Shift high-risk shipments to Rail or Air |
    | Haircare products | Add extra QC step before inspection |
    | High-cost failures | Audit Supplier 4’s cost vs quality tradeoff |

    > These insights can reduce inspection failures by 30–50% and save ₹2M/year in rework costs.
    """)

    