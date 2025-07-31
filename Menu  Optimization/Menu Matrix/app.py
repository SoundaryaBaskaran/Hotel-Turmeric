import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hotel Menu Engineering", layout="wide")

st.title("🏨 Hotel Menu Engineering Optimization App")
st.write("This app categorizes hotel dishes based on their profitability and popularity.")

# Load CSV directly from same folder
try:
    df = pd.read_csv("hotel_menu.csv")
except FileNotFoundError:
    st.error("❌ 'hotel_menu.csv' file not found. Please place the file in the same folder as this app.")
    st.stop()  # Stop execution if file not found

# Ensure correct column names
required_cols = {'Dish', 'Selling_Price', 'Cost_Price', 'Units_Sold'}
if not required_cols.issubset(set(df.columns)):
    st.error(f"CSV must include columns: {required_cols}")
    st.stop()

# Step 1: Compute Profit
df['Profit'] = df['Selling_Price'] - df['Cost_Price']

# Step 2: Averages for thresholding
avg_profit = df['Profit'].mean()
avg_units = df['Units_Sold'].mean()

# Step 3: Categorize each dish
def categorize(row):
    if row['Profit'] >= avg_profit and row['Units_Sold'] >= avg_units:
        return '⭐ Star'
    elif row['Profit'] >= avg_profit and row['Units_Sold'] < avg_units:
        return '🧩 Puzzle'
    elif row['Profit'] < avg_profit and row['Units_Sold'] >= avg_units:
        return '🐎 Plow Horse'
    else:
        return '🐶 Dog'

df['Category'] = df.apply(categorize, axis=1)

# Show categorized table
st.subheader("📋 Categorized Dishes")
st.dataframe(df.sort_values(by='Category'))

# Pie Chart of Category Distribution
st.subheader("📊 Category Distribution")
fig = px.pie(df, names='Category', title='Menu Item Categories')
st.plotly_chart(fig, use_container_width=True)

# Summary by Category
st.subheader("📌 Summary by Category")
summary = df.groupby('Category').agg(
    Total_Items=('Dish', 'count'),
    Total_Units_Sold=('Units_Sold', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Avg_Profit_Per_Item=('Profit', 'mean')
).reset_index()

# Format currency
summary['Total_Profit'] = summary['Total_Profit'].apply(lambda x: f"₹{x:,.2f}")
summary['Avg_Profit_Per_Item'] = summary['Avg_Profit_Per_Item'].apply(lambda x: f"₹{x:,.2f}")

st.dataframe(summary)

# Download Button
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Optimized Menu CSV", csv, "optimized_menu.csv", "text/csv")

