# data_ingestion_zack

Zack's data ingestion pipeline. Loads three car-price datasets, maps them to a common schema, and merges them into a single CSV.

## Script

- **merge_datasets.py** — Reads the three raw datasets, normalizes column names, and outputs `merged_output.csv` (~277K rows).

## Raw Data

| File | Source | Rows | Description |
|---|---|---|---|
| `huggingface_car_sales.csv` | HuggingFace | 10,000 | Make, model, year, engine size, mileage, fuel type, transmission, price |
| `uci_auto_imports.data` | UCI ML Repository | 205 | 1985 automobile specs (26 attributes, no header row) |
| `uci_auto_imports_codebook.txt` | UCI ML Repository | — | Column definitions and metadata for the UCI dataset |
| `uci_index.txt` | UCI ML Repository | — | Original file index from the UCI archive |
| `uci_width_horsepower_notes.txt` | — | — | Scratch notes on vehicle width vs. horsepower |
| `dvm_ad_table.csv` | DVM-CAR | 268K | Used-car ad listings (make, model, year, mileage, engine, gearbox, fuel, price) |
| `dvm_ad_table_extra.csv` | DVM-CAR | 268K | Duplicate/alternate version of the ad table |
| `dvm_basic_table.csv` | DVM-CAR | 1,011 | Automaker and model ID lookup table |
| `dvm_price_table.csv` | DVM-CAR | 6,333 | Entry prices by make, model, and year |

## Output

- **merged_output.csv** — Unified dataset with columns: `make`, `model`, `year`, `engine_size`, `mileage`, `fuel_type`, `transmission`, `price`, `source`
