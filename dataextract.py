import pandas as pd
import numpy as py
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

#defining columns to extract
#example
cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land' , 'wrong_fragment' , 'urgent' , 'hot','num_failed_logins','logged_in','num_compromised','root_shell' , 'su_attempted' ,'num_root' ,'num_file_creations' ,'num_shells' ,'num_access_files' ,'num_outbound_cmds','is_host_login','is_guest_login' ,'count','srv_count','serror_rate' ,'srv_serror_rate' ,'rerror_rate','srv_rerror_rate' , 'same_srv_rate', 'diff_srv_rate' , 'srv_diff_host_rate' , 'dst_host_count' ,'dst_host_srv_count' ,'dst_host_same_srv_rate','dst_host_diff_srv_rate' ,'dst_host_same_src_port_rate' , 'dst_host_srv_diff_host_rate' , 'dst_host_serror_rate' ,'dst_host_srv_serror_rate' , 'dst_host_rerror_rate' , 'dst_host_srv_rerror_rate' ,'attack_type', 'difficulty']

train = pd.read_csv("dataset/KDDTrain+.txt", names = cols)
test = pd.read_csv("dataset/KDDTest+.txt", names = cols)

attack_map = {
    'normal': 0, 'back': 1, 'buffer_overflow': 2, 'ftp_write': 3, 'guess_passwd': 4, 'imap': 5,
    'ipsweep': 6, 'land': 7, 'loadmodule': 8, 'multihop': 9, 'neptune': 10, 'nmap': 11,
    'perl': 12, 'phf': 13, 'pod': 14, 'portsweep': 15, 'rootkit': 16, 'satan': 17,
    'smurf': 18, 'spy': 19, 'teardrop': 20, 'warezclient': 21, 'warezmaster': 22
}

#map categorical columns to numeric values
pmap = {'icmp': 0, 'tcp': 1, 'udp': 2}
fmap = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10}

train['protocol_type'] = train['protocol_type'].map(pmap)
test['protocol_type'] = test['protocol_type'].map(pmap)

train['flag'] = train['flag'].map(fmap)
test['flag'] = test['flag'].map(fmap)

train.drop('service', axis = 1, inplace = True)
test.drop('service', axis = 1, inplace = True)

# Print the columns in the dataset to inspect them
#print(train.columns)

# If 'service' is not in the dataset, remove it
if 'service' not in train.columns:
    cols.remove('service')

#Map 'attack_type' to numerical values
train['attack_type'] = train['attack_type'].map(attack_map)
test['attack_type'] = test['attack_type'].map(attack_map)

#extract columns from training and test datasets
x_train_selected = train[cols]
x_test_selected = test[cols]

#target vars
y_train = train[['attack_type']]
y_test = test[['attack_type']]


# Drop rows with missing values
x_train_selected = pd.DataFrame(x_train_selected).dropna()
x_test_selected = pd.DataFrame(x_test_selected).dropna()

# Drop corresponding target rows as well to ensure consistency
y_train = y_train.loc[x_train_selected.index]
y_test = y_test.loc[x_test_selected.index]

#normalize, scale the data 
scale = MinMaxScaler()
x_train_selected = scale.fit_transform(x_train_selected)
x_test_selected = scale.transform(x_test_selected)

#train the gaussianNB model
clf = GaussianNB()
clf.fit(x_train_selected, y_train.values.ravel())

#predictions on test set
y_pred_selected = clf.predict(x_test_selected)


#predict and evaluate model performance
y_pred_selected = clf.predict(x_test_selected)
train_score = clf.score(x_train_selected, y_train)
test_score = clf.score(x_test_selected, y_test)

print(f"Training set R-squared score: {train_score}")
print(f"Test set R-squared score: {test_score}")

#save predictions to csv
#convert predictions and actual labels to a dataframe
predictions_df = pd.DataFrame({
    'Actual': y_test.values.ravel(),
    'Prediction': y_pred_selected
})

#save to csv
predictions_df.to_csv('predictions_selected_columns.csv', index=False)
print("Predictions have been saved to 'predictions_selected_columns.csv'")


