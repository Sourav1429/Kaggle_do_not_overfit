import pandas as pd
##training and testing paths
path_train="D://datasets//dont_overfit//train.csv"
path_test="D://datasets//dont_overfit//test.csv"
data=pd.read_csv(path_train)
#checking
data.head(10)
#dropping all Nan columns
data=data.dropna()
#importing Linear Discrimant analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#creating data for performing LDA
z=['target','id']
data_vars=data.columns.values.tolist()
X=[i for i in data_vars if i not in z]
y=['target']
#defining LDA function using solver as 'svd' shrinkage as No hence no change in data
#priors as None as well thus no specified shape, n_components is given as 2 so 2 dimensions from 300 dimensions
lda=LinearDiscriminantAnalysis(n_components=2)
X_r2=lda.fit_transform(data[X],data[y])
data=pd.read_csv(path_test)
data=data.dropna(axis=1)
X_test=[i for i in data_vars if i not in z]
pr=lda.predict(data[X_test])
new_dict={}
for i in data['id']:
    new_dict[i]=pr[int(i)-250]
new_arr=[['id','target']]
for i in data['id']:
    new_arr.append([i,new_dict[i]])
with open('final.csv','a') as f:
    writer=csv.writer(f)
    writer.writerows(new_arr)
f.close()
