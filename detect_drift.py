import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import logging
#detect drift code

# Setup logger to write logs to a file
logging.basicConfig(filename='drift_alert.log', level=logging.INFO)

# Load the datasets
old_data = pd.read_csv('data/employee_attrition.csv')
new_data = pd.read_csv('data/synthetic_employee_attrition.csv')

# List of features to check for drift
features_to_check = ['Age', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears', 'YearsInCurrentRole']

# Iterate through each feature and perform KS Test for drift detection
for feature in features_to_check:
    # KS Test to check drift between old and new data
    ks_stat, p_value = ks_2samp(old_data[feature], new_data[feature])
    
    # If p-value is less than 0.05, drift is detected
    if p_value < 0.05:
        logging.info(f"Data drift detected in '{feature}' feature. KS Test p-value: {p_value}")
    else:
        logging.info(f"No data drift detected in '{feature}' feature. KS Test p-value: {p_value}")

# You can also print an overall summary if required
logging.info("Drift detection completed.")
