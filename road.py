import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Remove or comment out the line causing the error
# st.set_option('deprecation.showPyplotGlobalUse', False) 
# This option is likely deprecated or renamed in newer Streamlit versions.

st.title("🚦 Road Accident Risk: Feature Selection + Model Comparison")

uploaded_file = st.file_uploader("📂 Upload your Excel dataset", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    data.dropna(inplace=True)

    # Encode categorical columns
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    if 'Accident_Risk_Level' not in data.columns:
        st.error("❗ Expected column 'Accident_Risk_Level' not found in dataset.")
    else:
        X = data.drop('Accident_Risk_Level', axis=1)
        y = data['Accident_Risk_Level']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initial Decision Tree for Feature Importance
        tree_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
        tree_model.fit(X_train, y_train)
        feature_importance = tree_model.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Information_Gain': feature_importance
        }).sort_values(by='Information_Gain', ascending=False)

        st.subheader("📌 Feature Importances (Information Gain)")
        st.dataframe(importance_df)

        threshold = st.slider("🎚 Select Information Gain Threshold", 0.0, 0.2, 0.05, 0.01)

        selected_features = importance_df[importance_df['Information_Gain'] > threshold]['Feature'].tolist()
        st.write(f"✅ Features Selected (IG > {threshold}):", selected_features)

        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }

        results = []

        for name, model in models.items():
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

        results_df = pd.DataFrame(results)
        st.subheader("📋 Model Comparison Table")
        st.dataframe(results_df)

        st.subheader("📈 Accuracy, Precision & Recall Comparison")
        plt.figure(figsize=(10, 6))
        plt.plot(results_df["Model"], results_df["Accuracy"], marker='o', label="Accuracy")
        plt.plot(results_df["Model"], results_df["Precision"], marker='s', label="Precision")
        plt.plot(results_df["Model"], results_df["Recall"], marker='^', label="Recall")
        plt.title("Model Comparison on Selected Features")
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        st.pyplot()