import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="UAE Student Support Dashboard", layout="wide")

st.title("UAE Student Support App Analytics Dashboard")
st.markdown("A data-driven dashboard for understanding international student needs in the UAE.")

# Load dataset
df = pd.read_csv("uae_student_support_dataset.csv")

# Sidebar
menu = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Profile",
        "EDA Visualisations",
        "Classification Analysis",
        "Regression Analysis",
        "Clustering Analysis",
        "Association Rules",
        "Business Recommendations"
    ]
)

# Helper copy
feature_yesno_cols = [
    "Need_Accommodation_Help",
    "Need_Food_Outlet_Info",
    "Need_Transport_Help",
    "Need_SIM_Card_Guidance",
    "Need_Currency_Exchange_Help",
    "Interested_in_Events",
    "Need_Supermarket_Info",
    "Interested_in_Places_to_Visit",
    "Interested_in_Online_Internships",
    "Interested_in_Part_Time_Jobs"
]

# Overview
if menu == "Overview":
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Avg Monthly Budget", f"AED {round(df['Monthly_Budget_AED'].mean(), 0)}")
    c4.metric("Avg Willingness to Pay", f"AED {round(df['Willingness_to_Pay_AED'].mean(), 1)}")

    st.write("### Dataset Preview")
    st.dataframe(df.head(10))

# Data Profile
elif menu == "Data Profile":
    st.subheader("Data Profile")
    st.write("### Shape")
    st.write(df.shape)

    st.write("### Data Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    st.write("### Missing Values")
    st.dataframe(pd.DataFrame(df.isnull().sum(), columns=["Missing Values"]))

    st.write("### Summary Statistics")
    st.dataframe(df.describe(include="all").transpose())

# EDA
elif menu == "EDA Visualisations":
    st.subheader("EDA Visualisations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Monthly Budget Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Monthly_Budget_AED"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("### Willingness to Pay Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Willingness_to_Pay_AED"], bins=20, kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### Likelihood to Use App")
    fig, ax = plt.subplots()
    df["Likelihood_to_Use_App"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.write("### Most Important Feature")
    fig, ax = plt.subplots(figsize=(10, 4))
    df["Most_Important_Feature"].value_counts().plot(kind="bar", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Classification
elif menu == "Classification Analysis":
    st.subheader("Classification Analysis")

    target = "Likelihood_to_Use_App"
    df_clf = df.copy()

    le_dict = {}
    for col in df_clf.columns:
        if df_clf[col].dtype == "object":
            le = LabelEncoder()
            df_clf[col] = le.fit_transform(df_clf[col])
            le_dict[col] = le

    X = df_clf.drop(columns=[target])
    y = df_clf[target]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []

    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    st.write("### Accuracy")
    st.write(f"Training Accuracy: {train_acc:.3f}")
    st.write(f"Testing Accuracy: {test_acc:.3f}")

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    if hasattr(model, "feature_importances_"):
        st.write("### Feature Importance")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        feat_imp.plot(kind="bar", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Regression
elif menu == "Regression Analysis":
    st.subheader("Regression Analysis")

    target = "Monthly_Budget_AED"
    df_reg = df.copy()

    for col in df_reg.columns:
        if df_reg[col].dtype == "object":
            le = LabelEncoder()
            df_reg[col] = le.fit_transform(df_reg[col])

    X = df_reg.drop(columns=[target])
    y = df_reg[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    st.write("### Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

# Clustering
elif menu == "Clustering Analysis":
    st.subheader("Clustering Analysis")

    df_cluster = df.copy()

    cluster_cols = [
        "Age",
        "Monthly_Budget_AED",
        "Willingness_to_Pay_AED"
    ] + feature_yesno_cols

    for col in cluster_cols:
        if df_cluster[col].dtype == "object":
            df_cluster[col] = df_cluster[col].map({"Yes": 1, "No": 0})

    X = df_cluster[cluster_cols].copy()
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    st.write("### Cluster Counts")
    st.dataframe(df_cluster["Cluster"].value_counts().reset_index())

    st.write("### Cluster Plot")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df_cluster["Monthly_Budget_AED"], df_cluster["Willingness_to_Pay_AED"], c=df_cluster["Cluster"])
    ax.set_xlabel("Monthly Budget AED")
    ax.set_ylabel("Willingness to Pay AED")
    st.pyplot(fig)

# Association Rules
elif menu == "Association Rules":
    st.subheader("Association Rules")

    assoc_df = df[feature_yesno_cols].copy()

    for col in assoc_df.columns:
        assoc_df[col] = assoc_df[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    assoc_df = assoc_df.fillna(0).astype(int)

    freq_items = apriori(assoc_df, min_support=0.1, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.5)

    if not rules.empty:
        show_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        rules_display = rules[show_cols].copy()
        rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(list(x)))
        st.dataframe(rules_display.sort_values(by="lift", ascending=False).head(10))
    else:
        st.write("No strong association rules found with current support/confidence thresholds.")


# Business Recommendations
elif menu == "Business Recommendations":
    st.subheader("Business Recommendations")

    st.markdown("""
    ### Key Business Recommendations

    1. **Prioritize accommodation and transport support**  
       These are likely to be high-demand pain points for new students.

    2. **Keep pricing affordable**  
       Students may have moderate monthly budgets but low willingness to pay for an information app, suggesting a low-cost or freemium model.

    3. **Bundle career features**  
       Internship and part-time job support can increase app relevance and engagement.

    4. **Segment users by need and budget**  
       Clustering can help design targeted offerings for different student groups.

    5. **Use association rules for feature bundling**  
       Frequently co-occurring needs can be turned into smart app bundles or onboarding journeys.
    """)
