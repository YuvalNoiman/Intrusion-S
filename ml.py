import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import Birch
from sklearn.ensemble import VotingClassifier

COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land' , 'wrong_fragment' , 'urgent' , 'hot','num_failed_logins','logged_in','num_compromised','root_shell' , 'su_attempted' ,'num_root' ,'num_file_creations' ,'num_shells' ,'num_access_files' ,'num_outbound_cmds','is_host_login','is_guest_login' ,'count','srv_count','serror_rate' ,'srv_serror_rate' ,'rerror_rate','srv_rerror_rate' , 'same_srv_rate', 'diff_srv_rate' , 'srv_diff_host_rate' , 'dst_host_count' ,'dst_host_srv_count' ,'dst_host_same_srv_rate','dst_host_diff_srv_rate' ,'dst_host_same_src_port_rate' , 'dst_host_srv_diff_host_rate' , 'dst_host_serror_rate' ,'dst_host_srv_serror_rate' , 'dst_host_rerror_rate' , 'dst_host_srv_rerror_rate' ,'attack_type', 'difficulty']

ATTACK_TYPES = {'ftp_write':'r2l', 'normal':'normal', 'rootkit':'u2r', 'imap':'r2l', 'ipsweep':'probe', 'nmap':'probe', 'loadmodule':'u2r', 'multihop':'r2l', 'neptune':'dos', 'teardrop':'dos', 'satan':'probe', 'land':'dos', 'phf':'r2l', 'warezmaster':'r2l', 'smurf':'dos', 'guess_passwd':'r2l', 'buffer_overflow':'u2r', 'perl':'u2r', 'portsweep':'probe', 'spy':'r2l', 'warezclient':'r2l', 'back':'dos', 'pod':'dos','saint':'probe','sqlattack':'u2r','mscan':'probe', 'apache2':'dos','snmpgetattack':'r2l','processtable':'dos','httptunnel':'u2r','ps':'u2r','snmpguess':'r2l','mailbomb':'dos','named':'r2l','sendmail':'r2l','xterm':'u2r','worm':'r2l','xlock':'r2l','xsnoop':'r2l','udpstorm':'dos'}

PMAP = {'icmp':0, 'tcp':1, 'udp':2}
FMAP = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}

AMAP = {'normal':0, 'r2l':1, 'u2r':2, 'probe':3, 'dos':4}


def ready_data(file_path):
	data = pd.read_csv(file_path, names = COLS)
	#print(data.head())
	#print(data.shape)
	#print(data.isnull().sum())
	#changes to 4 attack categories and normal
	#print(set(data["attack_type"]))
	data['attack_type'] = data.attack_type.apply(lambda r:ATTACK_TYPES[r[:]])
	#print(data.head())

	#num_cols = data._get_numeric_data().columns
	#categorical_cols = list(set(data.columns)-set(num_cols))
	#categorical_cols.remove('attack_type')
	#categorical_cols.remove('difficulty')

	#print(categorical_cols)

	# drop columns with NaN
	data = data.dropna(axis='columns')
	
	'''
	# keep columns where there are more than 1 unique values and are numeric
	unique_data = data[[col for col in data.columns if data[col].nunique() > 1 and pd.api.types.is_numeric_dtype(data[col])]]
	# Now calculate the correlation matrix
	corr = unique_data.corr()

	plt.figure(figsize =(15, 12))
	sns.heatmap(corr)
	plt.show()
	'''


	data.drop('num_root', axis = 1, inplace = True)

	data.drop('srv_serror_rate', axis = 1, inplace = True)

	data.drop('srv_rerror_rate', axis = 1, inplace = True)

	data.drop('dst_host_srv_serror_rate', axis = 1, inplace = True)

	data.drop('dst_host_serror_rate', axis = 1, inplace = True)

	data.drop('dst_host_rerror_rate', axis = 1, inplace = True)

	data.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True)

	data.drop('dst_host_same_srv_rate', axis = 1, inplace = True)
	
	data['protocol_type'] = data['protocol_type'].map(PMAP)
	data['flag'] = data['flag'].map(FMAP)

	data.drop('service', axis = 1, inplace = True)

	return data

def prediction_diff(y_pred, y_test):
	#print(len(y_pred), y_pred)

	li1 = np.array(y_test)
	li2 = np.array(y_pred)

	dif1 = np.setdiff1d(li1, li2)
	dif2 = np.setdiff1d(li2, li1)

	temp3 = np.concatenate((dif1, dif2))
	#print(list(temp3))
	#print(len(list(temp3)))
	print( str(len(list(temp3))) + " out of " + str(len(y_pred)) + " wrong!" )

def main(mode, alert_type):
	train = ready_data("/Users/yoavnoiman/Desktop/CPSC 454/Project/KDDTrain+.txt")
	x_train = train.drop(['attack_type'], axis = 1)
	y_train = train[['attack_type']].values.ravel()

	if mode == "1":
		print("Training Decision Tree")
		clf = DecisionTreeClassifier().fit(x_train, y_train)
	elif mode == "2":
		print("Training Neural Network")
		sc = MinMaxScaler()
		x_train = sc.fit_transform(x_train)
		clf = MLPClassifier(max_iter=1000).fit(x_train, y_train)	
	elif mode == "3":
		print("Training Outlier Detection and Clustering")
		qt = QuantileTransformer(output_distribution='normal')
		x_train = qt.fit_transform(x_train)
		ee = EllipticEnvelope(contamination=0.001, support_fraction=1).fit(x_train)
		brc = Birch(n_clusters=5).fit(x_train)
	else:
		print("Training Decision Tree and NN")
		clf1 = DecisionTreeClassifier()
		clf2 = MLPClassifier(max_iter=1000)
		vc = VotingClassifier(estimators=[('dtc', clf), ('nn', clf2)], voting='hard')
		vc = vc.fit(x_train, y_train)
		qt = QuantileTransformer(output_distribution='normal')
		x_train = qt.fit_transform(x_train)
		ee = EllipticEnvelope(contamination=0.001, support_fraction=1).fit(x_train)
		brc = Birch(n_clusters=5).fit(x_train)

	if alert_type == "1":
		while True:
			break
			
	if alert_type == "2":
		while True:
			break
	else:
		while True:
			break	
	print("Program ended")
		

if __name__ == "__main__":
	print("There are four modes: \n\nMode 1: Decision Tree \nMode 2: neural network \nMode 3: clustering and outlier detection \nMode 4: Combination of all 3!\n")
	modes = ["1","2","3","4"]
	while True:
		mode = input("Enter the number correlating to the mode you want: ")
		if mode not in modes:
			print("Try again")
		else:
			print("\n")
			break
	print("There are three ways to get alerted: \n\n(1) Machine side alerts \n(2) Server side alerts\n(3) Both machine side and server side alerts\n")
	while True:
		alert_type = input("Enter the number correlating to the alert you want: ")
		if alert_type not in modes[0:3]:
			print("Try again")
		else:
			print("\n")
			break
	main(mode, alert_type)