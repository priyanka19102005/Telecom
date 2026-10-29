import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Customer Churn Predictor & Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0369A1;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F0F9FF;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #0284C7;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Data Loading & Model Training (Cached)
# -------------------------------------------------------------
@st.cache_resource
def load_data_and_models():
    # Priority: 1. CSV in directory, 2. Excel in directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_candidates = [
        os.path.join(script_dir, "telecom_customer_churn.csv"),
        "telecom_customer_churn.csv",
        os.path.join(os.getcwd(), "telecom_customer_churn.csv")
    ]
    df = None
    for p in csv_candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break

    if df is None:
        excel_candidates = [
            os.path.join(script_dir, "telecom_customer_churn  LEC 8 DATA.xlsx"),
            "telecom_customer_churn  LEC 8 DATA.xlsx",
            os.path.join(os.getcwd(), "telecom_customer_churn  LEC 8 DATA.xlsx")
        ]
        for p in excel_candidates:
            if os.path.exists(p):
                df = pd.read_excel(p, sheet_name="telecom_customer_churn (4)")
                break

    if df is None:
        raise FileNotFoundError("Telecom dataset file not found in repository.")

    # Target variable handling
    if "Target" not in df.columns:
        if "Customer Status" in df.columns:
            df["Target"] = df["Customer Status"].apply(lambda s: 1 if str(s).strip().lower() == "churned" else 0)
        else:
            raise KeyError("Target or Customer Status column missing from dataset.")

    # Modeling features
    num_cols = [
        "Age", "Number of Dependents", "Number of Referrals",
        "Tenure in Months", "Monthly Charge", "Total Charges"
    ]
    cat_cols = [
        "Gender", "Married", "Phone Service", "Multiple Lines",
        "Internet Service", "Internet Type", "Online Security",
        "Online Backup", "Device Protection Plan", "Premium Tech Support",
        "Contract", "Paperless Billing", "Payment Method"
    ]

    # Fill NaNs for safety
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("No").astype(str)

    X = df[num_cols + cat_cols].copy()
    y = df["Target"].astype(int)

    # Preprocessors
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    models_dict = {
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
        ])
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    eval_results = []
    roc_data = {}

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        eval_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc,
            "Confusion Matrix": cm
        })
        roc_data[name] = (fpr, tpr, roc)

    eval_df = pd.DataFrame(eval_results)
    return df, X, y, models_dict, eval_df, roc_data, X_test, y_test, num_cols, cat_cols


# -------------------------------------------------------------
# Initialize Application
# -------------------------------------------------------------
try:
    df, X, y, models, eval_df, roc_data, X_test, y_test, num_cols, cat_cols = load_data_and_models()
except Exception as e:
    st.error(f"Error initializing Telecom Churn app: {e}")
    st.stop()

# -------------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3616/3616930.png", width=70)
st.sidebar.title("Telecom AI Hub")
st.sidebar.markdown("Predict customer churn probability and discover actionable retention strategies.")
st.sidebar.success(f"✓ Total Records: {len(df):,}")

churn_count = (df["Target"] == 1).sum()
churn_rate = (churn_count / len(df)) * 100
st.sidebar.metric("Overall Churn Rate", f"{churn_rate:.1f}%")

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">📡 Telecom Customer Churn Prediction & Retention Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify high-risk churn customers in advance, explore churn drivers, and automate targeted retention campaigns.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Customer Churn",
    "📊 Model Benchmarks & Drivers",
    "📈 Churn Drivers & Analytics",
    "📁 Batch Customer Scoring"
])

# =============================================================
# TAB 1: Real-Time Single Customer Prediction
# =============================================================
with tab1:
    st.subheader("Predict Churn Risk for an Individual Customer")

    c_m1, _ = st.columns([2, 2])
    with c_m1:
        chosen_model_name = st.selectbox(
            "Select Machine Learning Model",
            options=list(models.keys()),
            index=0,
            help="Random Forest provides the highest accuracy and balanced precision-recall."
        )

    st.markdown("---")

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Customer Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.slider("Age", 18, 95, 38)
            married = st.selectbox("Married?", ["Yes", "No"])
            dependents = st.number_input("Number of Dependents", 0, 10, 0)
            referrals = st.number_input("Number of Referrals", 0, 15, 0)

        with col2:
            st.markdown("##### 📱 Subscribed Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["Yes", "No"])
            internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])
            online_sec = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection Plan", ["No", "Yes"])
            tech_support = st.selectbox("Premium Tech Support", ["No", "Yes"])

        with col3:
            st.markdown("##### 💳 Contract & Account Details")
            tenure = st.slider("Tenure in Months", 1, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
            monthly_charge = st.number_input("Monthly Charge ($)", 10.0, 200.0, 75.0)
            total_charges = st.number_input("Total Charges ($)", 10.0, 10000.0, float(monthly_charge * tenure))

        submit_btn = st.form_submit_button("🔮 Predict Churn Probability", use_container_width=True)

    if submit_btn:
        customer_df = pd.DataFrame([{
            "Age": age,
            "Number of Dependents": dependents,
            "Number of Referrals": referrals,
            "Tenure in Months": tenure,
            "Monthly Charge": monthly_charge,
            "Total Charges": total_charges,
            "Gender": gender,
            "Married": married,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Internet Type": internet_type,
            "Online Security": online_sec,
            "Online Backup": online_backup,
            "Device Protection Plan": device_protection,
            "Premium Tech Support": tech_support,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment_method
        }])

        active_pipe = models[chosen_model_name]
        pred_churn = active_pipe.predict(customer_df)[0]
        prob_churn = float(np.clip(active_pipe.predict_proba(customer_df)[0][1], 0.0, 1.0))

        st.markdown("### 📋 Prediction Outcome")
        r1, r2 = st.columns([1, 1])

        with r1:
            if prob_churn >= 0.5:
                st.error(f"### ⚠️ High Churn Risk ({prob_churn * 100:.1f}%)")
                st.write("**Assessment**: This customer shows strong indicators of switching to another carrier.")
            elif prob_churn >= 0.3:
                st.warning(f"### ⚡ Moderate Churn Risk ({prob_churn * 100:.1f}%)")
                st.write("**Assessment**: The customer has some churn risk factors. Consider early retention engagement.")
            else:
                st.success(f"### ✅ Low Churn Risk ({prob_churn * 100:.1f}%)")
                st.write("**Assessment**: The customer is satisfied and likely to remain with the provider.")

        with r2:
            st.metric("Churn Risk Probability", f"{prob_churn * 100:.1f}%")
            st.progress(prob_churn)

        st.markdown("#### 💡 Recommended Retention Action Plan")
        if contract == "Month-to-Month":
            st.info("🎯 **Contract Upgrade Offer**: Customer is on a Month-to-Month plan. Offer a 10% discount on a 1-year contract.")
        if tech_support == "No":
            st.info("🛠️ **Support Add-on**: Customer lacks Premium Tech Support. Offer a complimentary 3-month trial.")
        if tenure <= 6:
            st.info("🎁 **New Customer Welcome**: Customer is in the high-risk onboarding phase (first 6 months). Send proactive check-in call.")


# =============================================================
# TAB 2: Model Comparison & Evaluation
# =============================================================
with tab2:
    st.subheader("Model Benchmark & Evaluation")
    st.write("Performance measured on a 20% holdout test dataset:")

    display_eval = eval_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].copy()
    st.dataframe(display_eval.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1 Score": "{:.2%}",
        "ROC-AUC": "{:.3f}"
    }), use_container_width=True)

    c_e1, c_e2 = st.columns(2)

    with c_e1:
        st.markdown("##### 📊 Accuracy & ROC-AUC Comparison")
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        melted = eval_df.melt(id_vars=["Model"], value_vars=["Accuracy", "ROC-AUC"], var_name="Metric", value_name="Score")
        sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax_bar, palette="Blues_d")
        ax_bar.set_ylim(0.5, 1.0)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with c_e2:
        st.markdown("##### 📈 ROC Curves")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        for m_name, (fpr, tpr, auc_val) in roc_data.items():
            ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random Chance")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.markdown("---")
    st.markdown("##### 🧮 Confusion Matrices")
    cm_cols = st.columns(len(eval_df))
    for i, r in eval_df.iterrows():
        with cm_cols[i]:
            st.markdown(f"**{r['Model']}**")
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(r["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                        xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)


# =============================================================
# TAB 3: Churn Analytics & Business Insights
# =============================================================
with tab3:
    st.subheader("Customer Churn Drivers & Behavioral Insights")

    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    col_k1.metric("Total Customers", f"{len(df):,}")
    col_k2.metric("Total Churned", f"{churn_count:,}")
    col_k3.metric("Churn Rate", f"{churn_rate:.1f}%")
    if "Monthly Charge" in df.columns:
        lost_rev = df[df["Target"] == 1]["Monthly Charge"].sum()
        col_k4.metric("Monthly Revenue Lost", f"${lost_rev:,.0f}/mo")

    st.markdown("---")
    c_a1, c_a2 = st.columns(2)

    with c_a1:
        st.markdown("##### Churn Reasons Breakdown")
        if "Churn Category" in df.columns:
            cat_df = df["Churn Category"].dropna().value_counts().reset_index()
            cat_df.columns = ["Category", "Count"]
            fig_pie = px.pie(cat_df, names="Category", values="Count", color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig_pie, use_container_width=True)

    with c_a2:
        st.markdown("##### Churn Rate by Contract Type")
        if "Contract" in df.columns:
            ctr_df = df.groupby("Contract")["Target"].mean().reset_index()
            ctr_df["Churn Rate (%)"] = (ctr_df["Target"] * 100).round(1)
            fig_ctr = px.bar(ctr_df, x="Contract", y="Churn Rate (%)", color="Contract", text="Churn Rate (%)",
                             color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"])
            fig_ctr.update_traces(textposition="outside")
            st.plotly_chart(fig_ctr, use_container_width=True)


# =============================================================
# TAB 4: Batch Customer Scoring
# =============================================================
with tab4:
    st.subheader("Batch Customer Churn Scoring")
    st.write("Upload a CSV file of customer records to score them with predicted churn risk:")

    batch_file = st.file_uploader("Upload Customer CSV for Batch Prediction", type=["csv"], key="telecom_batch")
    if batch_file is not None:
        try:
            b_df = pd.read_csv(batch_file)
            rf_model = models["Random Forest"]
            b_preds = rf_model.predict(b_df)
            b_probs = rf_model.predict_proba(b_df)[:, 1]

            out_df = b_df.copy()
            out_df["Churn_Prediction"] = np.where(b_preds == 1, "Churn Risk", "Loyal / Stay")
            out_df["Churn_Probability"] = np.round(b_probs, 4)
            out_df["Risk_Level"] = pd.cut(
                out_df["Churn_Probability"],
                bins=[-0.1, 0.3, 0.6, 1.0],
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )

            st.success(f"✓ Scored {len(out_df)} customer accounts successfully!")
            st.dataframe(out_df.head(25), use_container_width=True)

            csv_buf = io.StringIO()
            out_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Scored Customers CSV",
                data=csv_buf.getvalue(),
                file_name="telecom_churn_scored_customers.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as ex:
            st.error(f"Error processing batch scoring: {ex}")
