# Loads and merges our 3 car price datasets into one CSV
import pandas as pd
import os

# paths to raw data files
DATA_DIR = "data_raw"
OUTPUT_DIR = "data_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset 1: Hugging Face
print("Loading Hugging Face dataset")
df_hf = pd.read_csv(os.path.join(DATA_DIR, "huggingface_car_sales.csv"))
print(f"  {df_hf.shape[0]} rows, columns: {list(df_hf.columns)}")

hf = pd.DataFrame({
    "make": df_hf["Make"].str.strip().str.title(),
    "model": df_hf["Model"].str.strip(),
    "year": df_hf["Year"].astype(int),
    "engine_size": df_hf["Engine Size"].astype(float),
    "mileage": df_hf["Mileage"].astype(float),
    "fuel_type": df_hf["Fuel Type"].str.strip().str.title(),
    "transmission": df_hf["Transmission"].str.strip().str.title(),
    "price": df_hf["Price"].astype(float),
    "source": "huggingface"
})

# Dataset 2: UCI Automobile
# no header row in this file, column names come from imports-85.names
print("Loading UCI dataset")
uci_cols = [
    "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
    "num_of_doors", "body_style", "drive_wheels", "engine_location",
    "wheel_base", "length", "width", "height", "curb_weight",
    "engine_type", "num_of_cylinders", "engine_size", "fuel_system",
    "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
    "city_mpg", "highway_mpg", "price"
]
df_uci = pd.read_csv(os.path.join(DATA_DIR, "uci_auto_imports.data"), header=None, names=uci_cols, na_values="?")
df_uci = df_uci.dropna(subset=["price"])
print(f"  {df_uci.shape[0]} rows after dropping missing prices")

# UCI is missing year, mileage, model, and transmission columns
uci = pd.DataFrame({
    "make": df_uci["make"].str.strip().str.title(),
    "model": None,
    "year": None,
    "engine_size": pd.to_numeric(df_uci["engine_size"], errors="coerce"),
    "mileage": None,
    "fuel_type": df_uci["fuel_type"].str.strip().str.title(),
    "transmission": None,
    "price": pd.to_numeric(df_uci["price"], errors="coerce"),
    "source": "uci"
})

# Dataset 3: DVM-CAR Ad Table
print("Loading DVM-CAR dataset")
df_dvm = pd.read_csv(os.path.join(DATA_DIR, "dvm_ad_table.csv"), low_memory=False)
print(f"  {df_dvm.shape[0]} rows, columns: {list(df_dvm.columns)}")

# engine size has an 'L' suffix like "2.0L" that needs to be removed
eng = df_dvm["Engin_size"].astype(str).str.replace("L", "", regex=False)

dvm = pd.DataFrame({
    "make": df_dvm["Maker"].str.strip().str.title(),
    "model": df_dvm["Genmodel"].str.strip(),
    "year": pd.to_numeric(df_dvm["Reg_year"], errors="coerce"),
    "engine_size": pd.to_numeric(eng, errors="coerce"),
    "mileage": pd.to_numeric(df_dvm["Runned_Miles"], errors="coerce"),
    "fuel_type": df_dvm["Fuel_type"].str.strip().str.title(),
    "transmission": df_dvm["Gearbox"].str.strip().str.title(),
    "price": pd.to_numeric(df_dvm["Price"], errors="coerce"),
    "source": "dvm_car"
})

#Merge
print("\nMerging datasets")
merged = pd.concat([hf, uci, dvm], ignore_index=True)

# drop rows where price is missing or <= 0
merged = merged[merged["price"].notna() & (merged["price"] > 0)]

# reorder columns
merged = merged[["make", "model", "year", "engine_size", "mileage","fuel_type", "transmission", "price", "source"]]

# quick summary
print(f"\nMerged dataset: {len(merged):,} rows")
print(f"Rows per source:")
print(merged["source"].value_counts().to_string())
print(f"\nPrice range: ${merged['price'].min():,.0f} - ${merged['price'].max():,.0f}")
print(f"Year range: {merged['year'].dropna().min():.0f} - {merged['year'].dropna().max():.0f}")
print(f"Unique makes: {merged['make'].nunique()}")

# save it
out_path = os.path.join(OUTPUT_DIR, "merged_car_data.csv")
merged.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")