import os
import pandas as pd

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "states.csv") # add parent directory to path
# Load CSV into a DataFrame
stateDf = pd.read_csv(file_path, dtype={'FIPS2': str})

