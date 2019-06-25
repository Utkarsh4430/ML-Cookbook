import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings("ignore")
import random
from tqdm import tqdm
from functions_ALS import make_train,implicit_weighted_ALS,rec_items,get_items_purchased,calc_mean_auc,auc_score
import implicit
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# print("Reading Dataset")
# retail_data = pd.read_excel("./Online Retail.xlsx")
# print("done")
print("reading csv")
retail_data = pd.read_csv("./Online Retail.csv")
print("done")

#removing nan values
cleaned_retail = retail_data.loc[pd.isnull(retail_data.CustomerID) == False]

item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
item_lookup['StockCode'] = item_lookup.StockCode.astype(str) # Encode as strings for future lookup ease

cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to
# indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive

customers = list(np.sort(grouped_purchased.CustomerID.unique())) # Get our unique customers
products = list(grouped_purchased.StockCode.unique()) # Get our unique products that were purchased
quantity = list(grouped_purchased.Quantity) # All of our purchases

rows = grouped_purchased.CustomerID.astype('category', categories = customers).cat.codes 
# Get the associated row indices
cols = grouped_purchased.StockCode.astype('category', categories = products).cat.codes 
# Get the associated column indices
purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))

matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_purchases/matrix_size))

product_train, product_test, product_users_altered = make_train(purchases_sparse, pct_test = 0)

alpha = 15
model = implicit.als.AlternatingLeastSquares(factors=20,regularization = 0.1,iterations = 50)
# train the model on a sparse matrix of item/user/confidence weights
model.fit((product_train*alpha).astype('double'))
user_vecs = model.item_factors
item_vecs = model.user_factors

# calc_mean_auc(product_train, product_users_altered, 
#               [sparse.csr_matrix(user_vecs), sparse.csr_matrix(item_vecs.T)], product_test)
# AUC for our recommender system

customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix
products_arr = np.array(products) # Array of product IDs from the ratings matrix



for ix in range(5):
	user_input = random.choice(customers)
	print("Items bought:")
	print(get_items_purchased(user_input, product_train, customers_arr, products_arr, item_lookup))
	print("Recommended Items:")
	print(rec_items(user_input, product_train, user_vecs, item_vecs, customers_arr, products_arr, item_lookup,num_items = 10))
	print("-----------------------------------------------------")
	# print()