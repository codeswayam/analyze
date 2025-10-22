import json
import pandas as pd
import numpy as np # Required for np.nan

def main():
    # Read the data from 'data.csv'.
    # Ensuring compatibility with Python 3.11+ and Pandas 2.3.
    # Pandas 2.3 handles datetime parsing and general DataFrame operations efficiently.
    df = pd.read_csv("data.csv")

    # Compute the 'revenue' for each transaction as 'units' multiplied by 'price'.
    df["revenue"] = df["units"] * df["price"]

    # Convert the 'date' column to datetime objects.
    # This is crucial for time-based operations like rolling averages.
    df["date"] = pd.to_datetime(df["date"])

    # 1. Calculate `row_count`: total number of rows in the dataset.
    row_count = len(df)

    # 2. Calculate `regions_count`: count of distinct regions present in the data.
    regions_count = df["region"].nunique()

    # 3. Calculate `top_n_products_by_revenue`: a list of the top 3 products by total revenue.
    n = 3
    # Group data by 'product', sum their total 'revenue', sort in descending order,
    # and select the top N products.
    top_products = (
        df.groupby("product")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    # Format the top products into a list of dictionaries as required.
    # Revenues are rounded to 2 decimal places for cleaner output.
    top_products_list = [
        {"product": row["product"], "revenue": round(float(row["revenue"]), 2)}
        for _, row in top_products.iterrows()
    ]

    # 4. Calculate `rolling_7d_revenue_by_region`:
    # An object where keys are region names and values are the last 7-day rolling average of daily revenue.
    # First, calculate the sum of 'revenue' for each distinct 'region' and 'date' to get daily regional revenue.
    daily_region_revenue = (
        df.groupby(['region', 'date'])['revenue'].sum().reset_index()
    )

    # Set 'date' as the index for time-series operations, ensuring it's sorted by date.
    daily_region_revenue = daily_region_revenue.set_index('date').sort_index()

    rolling_7d_revenue_by_region = {}
    # Iterate through each distinct region to calculate its 7-day rolling average.
    for region_name, group in daily_region_revenue.groupby('region'):
        # For each region, calculate the 7-day moving average of its daily revenue.
        # '7D' specifies a 7-day calendar window.
        # 'min_periods=1' allows calculation even if fewer than 7 days of data are available in the window.
        rolling_avg = group['revenue'].rolling(window='7D', min_periods=1).mean()

        # Retrieve the last calculated value of this rolling average for the current region.
        # If the rolling_avg series is empty (e.g., no data for the region), assign NaN.
        last_rolling_value = rolling_avg.iloc[-1] if not rolling_avg.empty else np.nan

        # Convert any potential NaN values to None for JSON output, otherwise round to 2 decimal places.
        rolling_7d_revenue_by_region[region_name] = (
            round(float(last_rolling_value), 2) if pd.notna(last_rolling_value) else None
        )

    # Construct the final JSON object containing all calculated metrics.
    result_json = {
        "row_count": row_count,
        "regions_count": regions_count,
        "top_n_products_by_revenue": top_products_list,
        "rolling_7d_revenue_by_region": rolling_7d_revenue_by_region,
    }

    # Print the JSON object to stdout. `indent=2` makes the output human-readable.
    print(json.dumps(result_json, indent=2))

# Ensure the main function is called when the script is executed.
if __name__ == "__main__":
    main()
