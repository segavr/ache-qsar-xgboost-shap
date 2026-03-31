import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import os
import time

def fetch_data(target_id='CHEMBL220'):
    print(f"Fetching data for target {target_id}...")
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    
    all_data = []
    count = 0
    max_retries = 3
    
    # Use a list to store results and convert to DataFrame later
    print("Downloading data from ChEMBL in chunks...")
    
    # Instead of iterating over the whole result set at once, 
    # we can try to use slicing if the client supports it, or just be patient.
    # The error "fetch failed" suggests a network issue or timeout.
    
    try:
        for item in res:
            all_data.append(item)
            count += 1
            if count % 500 == 0:
                print(f"Downloaded {count} records...")
    except Exception as e:
        print(f"Error during download: {e}")
        if len(all_data) > 0:
            print(f"Saving partial data ({len(all_data)} records)...")
        else:
            raise e

    df = pd.DataFrame(all_data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/raw_data.csv', index=False)
    print(f"Total records saved: {len(df)}")

if __name__ == "__main__":
    fetch_data()
