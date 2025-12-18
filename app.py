import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Polynomial Regression App", layout="wide")
st.title("📈 Polynomial Regression App")

# Upload dataset
st.subheader("Upload CSV Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Select features
    st.subheader("Select Features")
    all_columns = df.columns.tolist()
    x_col = st.selectbox("Select Independent Variable (X)", all_columns)
    y_col = st.selectbox("Select Dependent Variable (Y)", all_columns)

    X = df[[x_col]].values
    y = df[y_col].values

    # Degree selection
    degree = st.slider("Select Polynomial Degree", 1, 5, 2)

    # Train-test split
    test_size = st.slider("Select Test Size (Fraction)", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Polynomial Features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    # Metrics
    st.subheader("Model Performance")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.3f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.3f}")
    st.write(f"**R² Score:** {r2:.3f}")

    # Plot
    st.subheader("📊 Regression Plot")
    X_range = np.linspace(X.min(), X.max(), 500).reshape(-1,1)
    y_range_pred = model.predict(poly.transform(X_range))

    plt.figure(figsize=(10,6))
    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X_range, y_range_pred, color="red", label=f"Polynomial Regression (Degree={degree})")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title("Polynomial Regression Fit")
    plt.legend()
    st.pyplot(plt)

    # Predict for new value
    st.subheader("🔮 Predict New Value")
    new_value = st.number_input(f"Enter value for {x_col}", float(X.min()), float(X.max()), float(X.mean()))
    if st.button("Predict"):
        pred = model.predict(poly.transform([[new_value]]))[0]
        st.success(f"Predicted {y_col} = {pred:.3f}")
