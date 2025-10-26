import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Fraudulent Provider Prediction App")

# =======================================
# STEP 1: TRAINING SECTION
# =======================================
st.header("Step 1: Upload Training Files")

# Upload training CSV files
train_beneficiary = st.file_uploader("Upload train_beneficiary.csv", type="csv")
train_inpatient = st.file_uploader("Upload train_inpatient.csv", type="csv")
train_outpatient = st.file_uploader("Upload train_outpatient.csv", type="csv")
train_main = st.file_uploader("Upload train.csv (Provider & Labels)", type="csv")

if train_beneficiary and train_inpatient and train_outpatient and train_main:
    # Load train data
    df_beneficiary = pd.read_csv(train_beneficiary)
    df_inpatient = pd.read_csv(train_inpatient)
    df_outpatient = pd.read_csv(train_outpatient)
    df_main = pd.read_csv(train_main)

    st.success("✅ Training files loaded successfully!")

    # -------------------------------
    # DATA MANAGEMENT & CLEANING
    # -------------------------------
    st.subheader("Step 1: Data Management & Cleaning")

    # Replace 'NA' and blank spaces with actual NaN
    for df in [df_beneficiary, df_inpatient, df_outpatient, df_main]:
        df.replace(['NA', 'NaN', ' ', '', '-'], pd.NA, inplace=True)

    def convert_numeric(df):
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        return df

    df_beneficiary = convert_numeric(df_beneficiary)
    df_inpatient = convert_numeric(df_inpatient)
    df_outpatient = convert_numeric(df_outpatient)

    def fill_missing(df):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('Unknown')
        return df

    df_beneficiary = fill_missing(df_beneficiary)
    df_inpatient = fill_missing(df_inpatient)
    df_outpatient = fill_missing(df_outpatient)

    st.success("✅ Missing values handled successfully!")

    # Chronic condition count
    chronic_cols = [c for c in df_beneficiary.columns if c.startswith('ChronicCond_')]
    df_beneficiary['ChronicCond_Count'] = df_beneficiary[chronic_cols].apply(lambda x: (x == 1).sum(), axis=1) if chronic_cols else 0

    # -------------------------------
    # FEATURE ENGINEERING
    # -------------------------------
    st.subheader("Step 2: Feature Engineering")

    # Convert date columns
    for df in [df_inpatient, df_outpatient]:
        for col in ['ClaimStartDt', 'ClaimEndDt']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Length of Stay
    if {'ClaimStartDt', 'ClaimEndDt'}.issubset(df_inpatient.columns):
        df_inpatient['LengthOfStay'] = (df_inpatient['ClaimEndDt'] - df_inpatient['ClaimStartDt']).dt.days.clip(lower=0)
    else:
        df_inpatient['LengthOfStay'] = 0

    # Diagnosis and procedure code counts
    diag_cols_in = [c for c in df_inpatient.columns if c.startswith('ClmDiagnosisCode_')]
    proc_cols_in = [c for c in df_inpatient.columns if c.startswith('ClmProcedureCode_')]
    diag_cols_out = [c for c in df_outpatient.columns if c.startswith('ClmDiagnosisCode_')]
    proc_cols_out = [c for c in df_outpatient.columns if c.startswith('ClmProcedureCode_')]

    df_inpatient['DiagCodeCount'] = df_inpatient[diag_cols_in].notna().sum(axis=1)
    df_inpatient['ProcCodeCount'] = df_inpatient[proc_cols_in].notna().sum(axis=1)
    df_outpatient['DiagCodeCount'] = df_outpatient[diag_cols_out].notna().sum(axis=1)
    df_outpatient['ProcCodeCount'] = df_outpatient[proc_cols_out].notna().sum(axis=1)

    # Aggregate provider-level metrics
    inpatient_extra = df_inpatient.groupby('Provider').agg({
        'LengthOfStay': 'mean',
        'DiagCodeCount': 'mean',
        'ProcCodeCount': 'mean'
    }).rename(columns={
        'LengthOfStay': 'Avg_Length_Of_Stay',
        'DiagCodeCount': 'Avg_DiagCode_Count_Inpatient',
        'ProcCodeCount': 'Avg_ProcCode_Count_Inpatient'
    })

    outpatient_extra = df_outpatient.groupby('Provider').agg({
        'DiagCodeCount': 'mean',
        'ProcCodeCount': 'mean'
    }).rename(columns={
        'DiagCodeCount': 'Avg_DiagCode_Count_Outpatient',
        'ProcCodeCount': 'Avg_ProcCode_Count_Outpatient'
    })

    df_model = (
        df_main
        .merge(inpatient_extra, on='Provider', how='left')
        .merge(outpatient_extra, on='Provider', how='left')
    )
    df_model.fillna(0, inplace=True)

    st.success("✅ Feature engineering completed!")

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    st.subheader("Step 3: Model Training & Evaluation")

    if 'PotentialFraud' not in df_model.columns:
        st.error("❌ Target column 'PotentialFraud' not found in train.csv.")
    else:
        X = df_model.drop(columns=['PotentialFraud', 'Provider'])
        y = df_model['PotentialFraud']

        # Encode categorical
        for col in X.select_dtypes(include='object').columns:
            X[col], _ = pd.factorize(X[col])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        st.markdown("### Classification Report")
        st.text(classification_report(y_test, y_pred))
        st.markdown(f"### ROC AUC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_[0]
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        st.dataframe(coef_df.head(15))

        st.success("✅ Model trained successfully!")

        # Store model & scaler
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['train_ready'] = True

        st.info("Now upload test files below to generate predictions 👇")

# =======================================
# STEP 2: TEST PREDICTION SECTION
# =======================================
st.header("Step 2: Upload Test Files for Predictions")

test_beneficiary = st.file_uploader("Upload test_beneficiary.csv", type="csv")
test_inpatient = st.file_uploader("Upload test_inpatient.csv", type="csv")
test_outpatient = st.file_uploader("Upload test_outpatient.csv", type="csv")
test_main = st.file_uploader("Upload test.csv (Provider only)", type="csv")

if test_beneficiary and test_inpatient and test_outpatient and test_main:
    if not st.session_state.get('train_ready', False):
        st.warning("⚠️ Please train the model first by uploading training files above.")
    else:
        model = st.session_state['model']
        scaler = st.session_state['scaler']

        st.success("✅ Test files uploaded successfully!")

        # Load test data
        df_bene_test = pd.read_csv(test_beneficiary)
        df_in_test = pd.read_csv(test_inpatient)
        df_out_test = pd.read_csv(test_outpatient)
        df_main_test = pd.read_csv(test_main)

        # Simple cleaning
        for df in [df_bene_test, df_in_test, df_out_test, df_main_test]:
            df.replace(['NA', 'NaN', ' ', '', '-'], pd.NA, inplace=True)
            df.fillna(0, inplace=True)

        # Simple example aggregation (reuse inpatient/outpatient logic)
        diag_cols_in = [c for c in df_in_test.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols_in = [c for c in df_in_test.columns if c.startswith('ClmProcedureCode_')]
        diag_cols_out = [c for c in df_out_test.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols_out = [c for c in df_out_test.columns if c.startswith('ClmProcedureCode_')]

        df_in_test['DiagCodeCount'] = df_in_test[diag_cols_in].notna().sum(axis=1)
        df_in_test['ProcCodeCount'] = df_in_test[proc_cols_in].notna().sum(axis=1)
        df_out_test['DiagCodeCount'] = df_out_test[diag_cols_out].notna().sum(axis=1)
        df_out_test['ProcCodeCount'] = df_out_test[proc_cols_out].notna().sum(axis=1)

        # Convert dates
        for col in ['ClaimStartDt', 'ClaimEndDt']:
            if col in df_in_test.columns:
                df_in_test[col] = pd.to_datetime(df_in_test[col], errors='coerce', dayfirst=True)

        # Compute LengthOfStay
        if {'ClaimStartDt', 'ClaimEndDt'}.issubset(df_in_test.columns):
            df_in_test['LengthOfStay'] = (df_in_test['ClaimEndDt'] - df_in_test['ClaimStartDt']).dt.days.clip(lower=0)
        else:
            df_in_test['LengthOfStay'] = 0


        inpatient_extra_test = df_in_test.groupby('Provider').agg({
            'LengthOfStay': 'mean',
            'DiagCodeCount': 'mean',
            'ProcCodeCount': 'mean'
        }).rename(columns={
            'LengthOfStay': 'Avg_Length_Of_Stay',
            'DiagCodeCount': 'Avg_DiagCode_Count_Inpatient',
            'ProcCodeCount': 'Avg_ProcCode_Count_Inpatient'
        })


        outpatient_extra_test = df_out_test.groupby('Provider').agg({
            'DiagCodeCount': 'mean', 'ProcCodeCount': 'mean'
        }).rename(columns={
            'DiagCodeCount': 'Avg_DiagCode_Count_Outpatient',
            'ProcCodeCount': 'Avg_ProcCode_Count_Outpatient'
        })

        df_model_test = (
            df_main_test
            .merge(inpatient_extra_test, on='Provider', how='left')
            .merge(outpatient_extra_test, on='Provider', how='left')
        ).fillna(0)

        X_test_final = df_model_test.drop(columns=['Provider'], errors='ignore')
        for col in X_test_final.select_dtypes(include='object').columns:
            X_test_final[col], _ = pd.factorize(X_test_final[col])

        # Align test features with training features
        X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)

        X_test_scaled = scaler.transform(X_test_final)
        test_pred = model.predict(X_test_scaled)
        test_prob = model.predict_proba(X_test_scaled)[:, 1]

        df_model_test['Predicted_Fraud'] = test_pred
        df_model_test['Fraud_Probability'] = test_prob

        st.subheader("Predictions Preview (Top 20)")
        st.dataframe(df_model_test[['Provider', 'Predicted_Fraud', 'Fraud_Probability']].head(20))

        csv = df_model_test.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Predictions CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )
