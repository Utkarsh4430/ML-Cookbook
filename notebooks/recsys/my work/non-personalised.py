import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
print("Reading Dataset...")
data = pd.read_csv("./Online Retail.csv")
print("Successful\n")


cleaned_retail = data.loc[pd.isnull(data.CustomerID) == False]
# cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity','Country']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['StockCode','Country']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to
# indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive

print("Choosing country...")
country = random.choice(np.unique(data.Country.values))
print("Country chosen: ",country,'\n')
print(grouped_purchased[grouped_purchased['Country']==country].sort_values('Quantity',ascending = False)[['StockCode','Quantity']].head(15))