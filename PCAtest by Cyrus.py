import pandas as pd
from pandas import ExcelFile
Testpath= '/Users/cyrus/MQ Data test import.xlsx'
DF = pd.read_excel(Testpath)
print(DF)