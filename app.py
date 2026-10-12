
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import shap


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "customer_churn_model.pkl"

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #10233f, #244b72);
            color: white;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            color: white;
            margin-bottom: 0.35rem;
            font-size: 2.2rem;
        }

        .hero p {
            color: #d9e7f5;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .section-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6483a5;
            font-weight: 700;
        }

        .risk-card {
            padding: 1.25rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            min-height: 150px;
        }

        .risk-value {
            font-size: 2.25rem;
            font-weight: 750;
            margin: 0.3rem 0;
        }

        .muted {
            color: #8290a2;
            font-size: 0.9rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 1rem;
            border-radius: 12px;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 9px;
            font-weight: 600;
            min-height: 2.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📊 ChurnIQ")
    st.caption("Customer retention intelligence")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Predict Churn", "About the Model"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**Model information**")
    st.caption("Dataset: Telco Customer Churn")
    st.caption("Task: Binary classification")
    st.caption("Output: Estimated churn probability")

    st.info(
        "Use predictions to support customer-retention "
        "decisions, not as a guarantee of future behavior."
    )

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Expected location: "
        "`models/customer_churn_model.pkl`"
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    
    import numpy as np
    import matplotlib.pyplot as plt

    def get_feature_importance(pipeline):
        """Extract feature importance from a fitted sklearn pipeline."""
        if not hasattr(pipeline, "steps"):
            return None

        estimator = pipeline.steps[-1][1]

        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importance = np.abs(estimator.coef_).mean(axis=0)
        else:
            return None

        # Recover feature names after preprocessing.
        try:
            preprocessor = pipeline[:-1]
            feature_names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            feature_names = [
                f"Feature {i + 1}" for i in range(len(importance))
            ]

        if len(feature_names) != len(importance):
            feature_names = [
                f"Feature {i + 1}" for i in range(len(importance))
            ]

        result = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance,
        })

        return result.sort_values("Importance", ascending=False)
    
    def explain_prediction(pipeline, customer):
        """Explain one prediction using SHAP on the transformed features."""
        preprocessor = pipeline[:-1]
        estimator = pipeline.steps[-1][1]

        transformed_customer = preprocessor.transform(customer)

        try:
            feature_names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            feature_names = [
                f"Feature {i + 1}"
                for i in range(transformed_customer.shape[1])
            ]

        # Use the estimator's output scale for the positive class.
        explainer = shap.Explainer(
            estimator,
            feature_names=feature_names,
        )

        shap_values = explainer(transformed_customer)

        classes = list(estimator.classes_)
        if "Yes" in classes:
            churn_index = classes.index("Yes")
        elif 1 in classes:
            churn_index = classes.index(1)
        else:
            raise ValueError(
                f"Could not identify the churn class. Classes: {classes}"
            )

        values = shap_values.values

        # Binary classification explainers may return a 3D array.
        if values.ndim == 3:
            values = values[0, :, churn_index]
        else:
            values = values[0]

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "SHAP value": values,
        })

        explanation["Impact"] = explanation["SHAP value"].map(
            lambda x: "Increases churn score" if x > 0
            else "Decreases churn score" if x < 0
            else "No effect"
        )

        return explanation.reindex(
            explanation["SHAP value"].abs()
            .sort_values(ascending=False)
            .index
        )


    #temporary line of code
    st.write("Model type:", type(model).__name__)
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="section-label">MACHINE LEARNING • CUSTOMER ANALYTICS</div>
        <h1>Customer Churn Prediction</h1>
        <p>
            Estimate customer churn risk and identify opportunities
            for proactive retention.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# ABOUT PAGE
# =========================================================
if page == "About the Model":
    st.subheader("About this application")

    st.write(
        "This application uses a trained machine learning pipeline "
        "to estimate whether a telecom customer may leave the service."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Problem type", "Classification")

    with col2:
        st.metric("Prediction target", "Customer churn")

    with col3:
        st.metric("Input features", "19")

    st.markdown("### How it works")

    st.markdown(
        """
        1. Enter a customer's account, service, and billing details.
        2. The saved pipeline preprocesses the input features.
        3. The trained model estimates the probability of churn.
        4. The application displays the estimate and suggested
           retention actions.
        """
    )

    st.warning(
        "The displayed risk categories are illustrative thresholds, "
        "not calibrated probabilities of business loss. Model quality "
        "depends on the training data and evaluation process."
    )

    st.markdown("### Global feature importance")

    importance_df = get_feature_importance(model)

    if importance_df is None:
        st.info(
            "This model does not expose native feature importance. "
            "We can use permutation importance as an alternative."
        )
    else:
        top_features = importance_df.head(10).sort_values(
            "Importance", ascending=True
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            top_features["Feature"],
            top_features["Importance"],
        )
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        ax.set_title("Top 10 influential features")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "For tree models, importance reflects the estimator's "
            "native feature-importance measure. For linear models, "
            "absolute coefficients are shown. These measures are not "
            "causal explanations and are not directly comparable."
        )

# =========================================================
# PREDICTION PAGE
# =========================================================
else:
    st.markdown("### Customer profile")
    st.caption(
        "Complete the sections below. Use the customer's actual "
        "account information whenever possible."
    )

    with st.form("customer_form"):
        st.markdown("#### 1. Account information")

        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with c2:
            partner = st.selectbox("Has a partner?", ["Yes", "No"])
            dependents = st.selectbox(
                "Has dependents?", ["Yes", "No"]
            )

        with c3:
            tenure = st.slider(
                "Tenure (months)", min_value=0, max_value=72, value=12
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

        st.divider()
        st.markdown("#### 2. Services")

        s1, s2, s3 = st.columns(3)

        with s1:
            phone_service = st.selectbox(
                "Phone service", ["Yes", "No"]
            )
            multiple_lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"],
            )
            internet_service = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
            )

        with s2:
            online_security = st.selectbox(
                "Online security",
                ["No", "Yes", "No internet service"],
            )
            online_backup = st.selectbox(
                "Online backup",
                ["No", "Yes", "No internet service"],
            )
            device_protection = st.selectbox(
                "Device protection",
                ["No", "Yes", "No internet service"],
            )

        with s3:
            tech_support = st.selectbox(
                "Tech support",
                ["No", "Yes", "No internet service"],
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                ["No", "Yes", "No internet service"],
            )

        st.divider()
        st.markdown("#### 3. Billing and payment")

        b1, b2, b3 = st.columns(3)

        with b1:
            paperless_billing = st.selectbox(
                "Paperless billing", ["Yes", "No"]
            )
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        with b2:
            monthly_charges = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=2000.0,
                value=70.0,
                step=1.0,
            )

        with b3:
            total_charges = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=840.0,
                step=10.0,
                help="Enter the customer's accumulated charges.",
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Customer",
            use_container_width=True,
            type="primary",
        )

    # =====================================================
    # PREDICTION AND RESULTS
    # =====================================================
    if submitted:
        # Basic consistency checks
        if phone_service == "Yes" and multiple_lines == "No phone service":
            st.error(
                "Multiple Lines cannot be 'No phone service' when "
                "Phone Service is Yes."
            )
            st.stop()

        if phone_service == "No" and multiple_lines != "No phone service":
            st.error(
                "When Phone Service is No, Multiple Lines should be "
                "'No phone service'."
            )
            st.stop()

        internet_features = [
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]

        if internet_service == "No" and any(
            value != "No internet service"
            for value in internet_features
        ):
            st.error(
                "When Internet Service is No, set all internet-related "
                "features to 'No internet service'."
            )
            st.stop()

        if internet_service != "No" and any(
            value == "No internet service"
            for value in internet_features
        ):
            st.error(
                "For a customer with internet service, select Yes or No "
                "for each internet-related feature."
            )
            st.stop()

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        try:
            probabilities = model.predict_proba(customer)[0]

            # Find the probability associated with the "Yes" class.
            
# Find the probability corresponding to the churn class.
            classes = list(model.classes_)

            if "Yes" in classes:
                churn_index = classes.index("Yes")
            elif 1 in classes:
                churn_index = classes.index(1)
            elif True in classes:
                churn_index = classes.index(True)
            else:
                st.error(
                    f"Could not identify the churn class. Model classes: {classes}. "
                    "Check how the target was encoded during training."
                )
                st.stop()

            churn_probability = float(probabilities[churn_index])


        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        # Prediction threshold remains 0.50.
        prediction = int(churn_probability >= 0.50)

        # Illustrative risk bands, independent of the decision threshold.
        if churn_probability < 0.30:
            risk_level = "Low"
            risk_message = "Lower estimated churn risk"
        elif churn_probability < 0.60:
            risk_level = "Medium"
            risk_message = "Moderate estimated churn risk"
        else:
            risk_level = "High"
            risk_message = "Higher estimated churn risk"

        st.divider()
        st.markdown("## Analysis results")

        result1, result2, result3 = st.columns(3)

        with result1:
            st.metric(
                "Estimated churn probability",
                f"{churn_probability:.1%}",
            )

        with result2:
            st.metric("Risk category", risk_level)

        with result3:
            st.metric(
                "Model classification",
                "Likely to churn" if prediction else "Likely to stay",
            )

        st.progress(churn_probability)
        st.caption(
            f"{risk_message}. Risk bands are illustrative; the model "
            "probability has not necessarily been calibrated."
        )

        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Prediction summary")

            if prediction:
                st.error(
                    "⚠️ The model classifies this customer as likely to churn "
                    "at the current 0.50 decision threshold."
                )
            else:
                st.success(
                    "✅ The model classifies this customer as likely to stay "
                    "at the current 0.50 decision threshold."
                )

            st.markdown("**Customer snapshot**")
            st.write(f"- Tenure: {tenure} months")
            st.write(f"- Contract: {contract}")
            st.write(f"- Monthly charges: ${monthly_charges:,.2f}")
            st.write(f"- Internet service: {internet_service}")
            st.write(f"- Payment method: {payment_method}")

        with right:
            st.markdown("### Suggested next steps")

            if prediction:
                st.write(
                    "Consider a retention review before taking action."
                )

                if contract == "Month-to-month":
                    st.write(
                        "- Review whether a longer-term plan offers "
                        "meaningful value to this customer."
                    )

                if tenure < 12:
                    st.write(
                        "- Consider an onboarding check-in to identify "
                        "early service or billing concerns."
                    )

                if tech_support == "No" and internet_service != "No":
                    st.write(
                        "- Offer help with setup or technical issues "
                        "if relevant to the customer's experience."
                    )

                if payment_method == "Electronic check":
                    st.write(
                        "- Review billing experience and available "
                        "payment options with the customer."
                    )

                st.caption(
                    "These are general business-rule suggestions, not "
                    "explanations of why the model made this prediction."
                )

            else:
                st.write(
                    "Continue delivering reliable service and monitor "
                    "customer feedback."
                )
                st.write(
                    "- Maintain clear billing and support communication."
                )
                st.write(
                    "- Reassess risk when account or service details change."
                )

        st.caption(
            "Important: This output is a statistical estimate, not a "
            "guarantee. Risk bands and recommendations are illustrative. "
            "Use predictions alongside customer feedback and business "
            "judgment, not as the sole basis for decisions."
        )
        
        st.divider()
        st.markdown("### Why did the model make this prediction?")

        with st.spinner("Calculating explanation..."):
            try:
                explanation_df = explain_prediction(model, customer)
                top_explanations = explanation_df.head(10).copy()

                st.caption(
                    "Positive SHAP values push the model output toward churn; "
                    "negative values push it away from churn. The exact output "
                    "scale depends on the estimator and explainer."
                )

                st.dataframe(
                    top_explanations,
                    use_container_width=True,
                    hide_index=True,
                )

                chart_data = top_explanations.sort_values("SHAP value")

                fig, ax = plt.subplots(figsize=(9, 5))
                bar_colors = [
                    "#d9534f" if value > 0 else "#278a68"
                    for value in chart_data["SHAP value"]
                ]

                ax.barh(
                    chart_data["Feature"],
                    chart_data["SHAP value"],
                    color=bar_colors,
                )
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("SHAP value")
                ax.set_title("Top factors influencing this prediction")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

            except Exception as exc:
                st.warning(
                    "The prediction succeeded, but SHAP could not explain it "
                    f"with the current explainer: {exc}"
                )


    else:
        st.info(
            "Enter the customer's details above and select "
            "**Analyze Customer** to view the prediction."
        )
