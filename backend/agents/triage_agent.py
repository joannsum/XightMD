#triage

# Login using e.g. `huggingface-cli login` to access this dataset
from huggingface_hub import login
login(token="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")

import pandas as pd
from datasets import load_dataset

ds = load_dataset("rajpurkarlab/ReXGradient-160K")

# Convert the 'train' split to a DataFrame and print the first 5 rows
df = ds["train"].to_pandas()
print(df.head())