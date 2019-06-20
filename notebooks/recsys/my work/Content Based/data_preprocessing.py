import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
# from tqdm import tqdm
warnings.filterwarnings("ignore")

# reading the product catalog
data = pd.read_csv("Products.csv",sep = ';',encoding = 'unicode_escape')

#removing duplicate columns
data = data.drop_duplicates(keep = False).reset_index(drop=True)

#removing NULL values
data = data.loc[pd.isnull(data.VendorSubdepartment) == False]

#creating similarity matrix
x = data.values[:,2:5]
bag_of_words = []
for i in x:
    a = " "
    i[0] = "_".join(str(i[0]).split())
    a += i[0].lower()
    a+=" "
    i[1] = "_".join(str(i[1]).split())
    a+= i[1].lower()
    
    for j in str(i[2]).split(','):
        a+= " "
        j = "_".join(j.strip().split())
        a+= str(j).lower()
    a = " ".join(list(set(a.strip().split())))
    bag_of_words.append(a.strip())

count = CountVectorizer()
count_matrix = count.fit_transform(bag_of_words)
count_matrix = normalize(count_matrix, norm='l1', axis=1)
indices = pd.Series(data.VendorDescription)
data1 = data.values

