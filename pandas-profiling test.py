# This is a sample Python script.
from pandas_profiling import ProfileReport
import pandas as pd
df = pd.read_csv("2016 Crime Data.csv")

prof = ProfileReport(df)
prof.to_file(output_file= 'output.html')