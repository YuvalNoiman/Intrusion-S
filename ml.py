import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land' , 'wrong_fragment' , 'urgent' , 'hot','num_failed_logins','logged_in','num_compromised','root_shell' , 'su_attempted' ,'num_root' ,'num_file_creations' ,'num_shells' ,'num_access_files' ,'num_outbound_cmds','is_host_login','is_guest_login' ,'count','srv_count','serror_rate' ,'srv_serror_rate' ,'rerror_rate','srv_rerror_rate' , 'same_srv_rate', 'diff_srv_rate' , 'srv_diff_host_rate' , 'dst_host_count' ,'dst_host_srv_count' ,'dst_host_same_srv_rate','dst_host_diff_srv_rate' ,'dst_host_same_src_port_rate' , 'dst_host_srv_diff_host_rate' , 'dst_host_serror_rate' ,'dst_host_srv_serror_rate' , 'dst_host_rerror_rate' , 'dst_host_srv_rerror_rate' ,'attack_type', 'difficulty']
attack_types = {'ftp_write':'r2l', 'normal':'normal', 'rootkit':'u2r', 'imap':'r2l', 'ipsweep':'probe', 'nmap':'probe', 'loadmodule':'u2r', 'multihop':'r2l', 'neptune':'dos', 'teardrop':'dos', 'satan':'probe', 'land':'', 'phf':'r2l', 'warezmaster':'r2l', 'smurf':'dos', 'guess_passwd':'r2l', 'buffer_overflow':'u2r', 'perl':'u2r', 'portsweep':'probe', 'spy':'r2l', 'warezclient':'r2l', 'back':'dos', 'pod':'dos','saint':'probe','sqlattack':'u2r','mscan':'probe', 'apache2':'dos','snmpgetattack':'r2l','processtable':'dos','httptunnel':'u2r','ps':'u2r','snmpguess':'r2l','mailbomb':'dos','named':'r2l','sendmail':'r2l','xterm':'u2r','worm':'r2l','xlock':'r2l','xsnoop':'r2l','udpstorm':'dos'}

train = pd.read_csv("/Users/yoavnoiman/Desktop/CPSC 454/Project/KDDTrain+.txt", names = cols)
test = pd.read_csv("/Users/yoavnoiman/Desktop/CPSC 454/Project/KDDTest+.txt", names = cols)
print(test.head())
print(train.shape)
print(train.isnull().sum())

#changes to 4 attack categories and normal
print(set(train["attack_type"]))
print(set(test["attack_type"]))
train['attack_type'] = train.attack_type.apply(lambda r:attack_types[r[:]])
print(train.head())
test['attack_type'] = test.attack_type.apply(lambda r:attack_types[r[:]])
print(test.head())

num_cols = train._get_numeric_data().columns
categorical_cols = list(set(train.columns)-set(num_cols))
categorical_cols.remove('attack_type')
#categorical_cols.remove('difficulty')

print(categorical_cols)

# drop columns with NaN
train = train.dropna(axis='columns')
test = test.dropna(axis='columns')

'''
# keep columns where there are more than 1 unique values and are numeric
ntrain = train[[col for col in train.columns if train[col].nunique() > 1 and pd.api.types.is_numeric_dtype(train[col])]]

# Now calculate the correlation matrix
corr = ntrain.corr()

plt.figure(figsize =(15, 12))
sns.heatmap(corr)
#plt.show()'''


train.drop('num_root', axis = 1, inplace = True)

train.drop('srv_serror_rate', axis = 1, inplace = True)

train.drop('srv_rerror_rate', axis = 1, inplace = True)

train.drop('dst_host_srv_serror_rate', axis = 1, inplace = True)

train.drop('dst_host_serror_rate', axis = 1, inplace = True)

train.drop('dst_host_rerror_rate', axis = 1, inplace = True)

train.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True)

train.drop('dst_host_same_srv_rate', axis = 1, inplace = True)

test.drop('num_root', axis = 1, inplace = True)

test.drop('srv_serror_rate', axis = 1, inplace = True)

test.drop('srv_rerror_rate', axis = 1, inplace = True)

test.drop('dst_host_srv_serror_rate', axis = 1, inplace = True)

test.drop('dst_host_serror_rate', axis = 1, inplace = True)

test.drop('dst_host_rerror_rate', axis = 1, inplace = True)

test.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True)

test.drop('dst_host_same_srv_rate', axis = 1, inplace = True)

pmap = {'icmp':0, 'tcp':1, 'udp':2}
train['protocol_type'] = train['protocol_type'].map(pmap)
test['protocol_type'] = test['protocol_type'].map(pmap)
fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}
train['flag'] = train['flag'].map(fmap)
test['flag'] = test['flag'].map(fmap)

train.drop('service', axis = 1, inplace = True)
test.drop('service', axis = 1, inplace = True)


# Target variable and train set
y_train = train[['attack_type']]
x_train = train.drop(['attack_type'], axis = 1)
y_test = test[['attack_type']]
x_test = test.drop(['attack_type'], axis = 1)

sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(x_train, y_train.values.ravel())
y_pred = clf.predict(x_test)
print("R-squared score", clf.score(x_train, y_train))
print("R-squared score", clf.score(x_test, y_test))
#print("R-squared score", clf.score(list(y_pred), list(y_test)))
print(x_test)
print(y_test)
print(len(y_pred), y_pred)

li1 = np.array(y_test)
li2 = np.array(y_pred)

dif1 = np.setdiff1d(li1, li2)
dif2 = np.setdiff1d(li2, li1)

temp3 = np.concatenate((dif1, dif2))
print(list(temp3))
print(len(list(temp3)))
