!pip install streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

st.title("Road Accident Risk - Feature Selection using Information Gain")

uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    data.dropna(inplace=True)

    # Encode categorical columns
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    if 'Accident_Risk_Level' not in data.columns:
        st.error("Expected column 'Accident_Risk_Level' not found in dataset.")
    else:
        X = data.drop('Accident_Risk_Level', axis=1)
        y = data['Accident_Risk_Level']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(X_train, y_train)

        feature_importance = model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Information_Gain': feature_importance
        }).sort_values(by='Information_Gain', ascending=False)

        st.subheader("Feature Importances (Information Gain)")
        st.dataframe(importance_df)

        thresholds = [0.0, 0.01, 0.05, 0.1, 0.15]
        results = []

        for threshold in thresholds:
            selected_features = importance_df[importance_df['Information_Gain'] > threshold]['Feature']
            if selected_features.empty:
                continue
            X_train_sel = X_train[selected_features]
            X_test_sel = X_test[selected_features]

            model_sel = DecisionTreeClassifier(random_state=42)
            model_sel.fit(X_train_sel, y_train)
            y_pred = model_sel.predict(X_test_sel)

            results.append({
                'Threshold': threshold,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro'),
                'Recall': recall_score(y_test, y_pred, average='macro')
            })

        if results:
            results_df = pd.DataFrame(results)

            st.subheader("Model Performance vs. Information Gain Threshold")
            st.dataframe(results_df)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(results_df['Threshold'], results_df['Accuracy'], marker='o', label='Accuracy')
            ax.plot(results_df['Threshold'], results_df['Precision'], marker='s', label='Precision')
            ax.plot(results_df['Threshold'], results_df['Recall'], marker='^', label='Recall')
            ax.set_xlabel("Information Gain Threshold")
            ax.set_ylabel("Score")
            ax.set_title("Model Performance vs. Threshold")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)
        else:
            st.warning("No features selected for any threshold.")
