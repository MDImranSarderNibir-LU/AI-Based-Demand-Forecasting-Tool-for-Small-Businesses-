# AI-Based Demand Forecasting Tool for Small Businesses

This software solves the real-life project problem by helping a small business forecast future product demand from historical sales data.

## What it does
- Upload CSV or Excel sales data
- Clean missing values and outliers
- Aggregate demand by day, week, or month
- Compare Linear Regression, Random Forest, and Holt-Winters models
- Select the best model using MAE and RMSE
- Show forecast charts
- Export forecast results to CSV

## Run the software
1. Install Python 3.10 or newer
2. Open terminal in this folder
3. Install packages:
   pip install -r requirements.txt
4. Start the app:
   streamlit run app.py

## Input format
Your file should contain:
- one date column
- one sales or demand column
- optional product/category column
