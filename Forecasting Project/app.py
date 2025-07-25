import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Hotel Kitchen Forecasting Dashboard")

# File upload for dynamic data
uploaded_file = st.file_uploader("merged_data.csv", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Slider for forecast period (outside conditional to persist)
    forecast_period = st.slider("Select Forecast Period (Days)", min_value=1, max_value=365, value=30, step=1)

    # Total Demand Forecast
    st.subheader(f"Total Demand Forecast (Next {forecast_period} Days)")  
    daily_demand = df.groupby('Order_Date')['Quantity_Ordered'].sum().reset_index()
    forecast_df = daily_demand.rename(columns={'Order_Date': 'ds', 'Quantity_Ordered': 'y'})
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    model_total = Prophet()
    model_total.fit(forecast_df)
    future_total = model_total.make_future_dataframe(periods=forecast_period)
    forecast_total = model_total.predict(future_total)
    
    # Plot with red line (only predictions, improved clarity)
    fig, ax = plt.subplots(figsize=(10, 6))  # Increased figure size
    ax.plot(forecast_total['ds'], forecast_total['yhat'], 'g-', label='Predicted Data', linewidth=2)  # Thicker line
    last_train_date = forecast_df['ds'].max()
    ax.axvline(x=last_train_date, color='red', linestyle='--', label='Forecast Start', linewidth=2)
    ax.set_title(f"Total Demand Forecast (Next {forecast_period} Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Demand")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)  # Added grid for reference
    st.pyplot(fig)
    st.write(f"Average Predicted Demand: {forecast_total['yhat'].tail(forecast_period).mean():.2f} orders/day")

    # Dish-Specific Forecast
    st.subheader(f"Dish-Specific Forecast (Next {forecast_period} Days)")  # Fixed string formatting
    selected_dishes = ['Chicken Biryani', 'Paneer Butter Masala', 'Masala Dosa', 'Chicken Fried Rice', 'Chilly Chicken', 'Gobi Manchurian']
    df_filtered = df[df['Dish_Name'].isin(selected_dishes)]
    df_grouped = df_filtered.groupby(['Order_Date', 'Dish_Name'])['Quantity_Ordered'].sum().reset_index()
    df_grouped['Order_Date'] = pd.to_datetime(df_grouped['Order_Date'])
    dish = st.selectbox("Select Dish", selected_dishes)
    df_dish = df_grouped[df_grouped['Dish_Name'] == dish].copy()
    df_dish = df_dish.rename(columns={'Order_Date': 'ds', 'Quantity_Ordered': 'y'})
    model_dish = Prophet()
    model_dish.fit(df_dish)
    future_dish = model_dish.make_future_dataframe(periods=forecast_period)
    forecast_dish = model_dish.predict(future_dish)
    
    # Plot with red line (only predictions, improved clarity)
    fig, ax = plt.subplots(figsize=(10, 6))  # Increased figure size
    ax.plot(forecast_dish['ds'], forecast_dish['yhat'], 'g-', label='Predicted Data', linewidth=2)  # Thicker line
    last_train_date = df_dish['ds'].max()
    ax.axvline(x=last_train_date, color='red', linestyle='--', label='Forecast Start', linewidth=2)
    ax.set_title(f"{dish} Forecast (Next {forecast_period} Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)  # Added grid for reference
    st.pyplot(fig)
    avg_qty = forecast_dish[forecast_dish['ds'] > df_dish['ds'].max()]['yhat'].mean()
    st.write(f"Average Predicted Quantity for {dish}: {round(avg_qty)} orders")

    # Bar Chart for All Dishes
    st.subheader(f"Average Forecasted Demand per Dish (Next {forecast_period} Days)")  # Fixed string formatting
    summary = []
    for dish in selected_dishes:
        df_dish = df_grouped[df_grouped['Dish_Name'] == dish].copy()
        df_dish = df_dish.rename(columns={'Order_Date': 'ds', 'Quantity_Ordered': 'y'})
        model_dish = Prophet()
        model_dish.fit(df_dish)
        future_dish = model_dish.make_future_dataframe(periods=forecast_period)
        forecast_dish = model_dish.predict(future_dish)
        avg_qty = forecast_dish[forecast_dish['ds'] > df_dish['ds'].max()]['yhat'].mean()
        summary.append({'Dish': dish, 'Avg_Predicted_Qty': round(avg_qty)})
    forecast_summary_df = pd.DataFrame(summary).sort_values(by='Avg_Predicted_Qty', ascending=False)
    st.bar_chart(forecast_summary_df.set_index('Dish')['Avg_Predicted_Qty'])

# Run: streamlit run app.py