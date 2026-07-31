import pandas as pd
from feature_pipeline import fetch_historical_weather, add_derived_features, write_to_feature_store
pm25_df=pd.read_csv("data/raw/islamabad_pm25_historical.csv")
pm25_df.columns=pm25_df.columns.str.strip()
pm25_df["date"]=pd.to_datetime(pm25_df["date"])
pm25_df=pm25_df.sort_values("date").reset_index(drop=True)
start_date=pm25_df["date"].min().strftime("%Y-%m-%d")
end_date=pm25_df["date"].max().strftime("%Y-%m-%d")
print(f"Fetching historical weather from {start_date} to {end_date}...")
weather_df=fetch_historical_weather(start_date, end_date)
merged=pd.merge(pm25_df,weather_df,on="date",how="inner")
merged=merged.rename(columns={"date":"timestamp"})
merged["hour"]=12  
merged["day"]=merged["timestamp"].dt.day.astype("int64")
merged["month"]=merged["timestamp"].dt.month.astype("int64")
merged["pressure"]=merged["pressure"].astype("int64")
print(f"Merged {len(merged)} rows of historical features.")
merged=add_derived_features(merged)
BATCH_SIZE=500
for i in range(0, len(merged),BATCH_SIZE):
    batch=merged.iloc[i:i+BATCH_SIZE]
    print(f"Uploading rows {i} to {i+len(batch)}...")
    write_to_feature_store(batch)
print("Backfill complete.")