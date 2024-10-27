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
		
		
		

def test():
	# Target variable and train set
	train = ready_data("/Users/yoavnoiman/Desktop/CPSC 454/Project/KDDTrain+.txt")
	test= ready_data("/Users/yoavnoiman/Desktop/CPSC 454/Project/KDDTest+.txt")
	y_train = train[['attack_type']].values.ravel()
	x_train = train.drop(['attack_type'], axis = 1)
	y_test = test[['attack_type']].values.ravel()
	x_test = test.drop(['attack_type'], axis = 1)
	#####

	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)

	print("Ridge")
	from sklearn.linear_model import RidgeClassifier
	clf = RidgeClassifier().fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)
		
	#print("SVM")
	#from sklearn import svm
	#clf = svm.SVC()
	#clf.fit(x_train, y_train)
	#print(clf.score(x_train, y_train))

	print("SDGC")
	from sklearn.linear_model import SGDClassifier
	clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
	clf.fit(x_train, y_train) 
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)

	#print("Neighbors")
	#from sklearn.neighbors import NearestNeighbors
	#nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(x_train)
	#distances, indices = nbrs.kneighbors(x_train)

	#print("GPC")
	#from sklearn.gaussian_process import GaussianProcessClassifier
	#from sklearn.gaussian_process.kernels import RBF
	#kernel = 1.0 * RBF(1.0)
	#gpc = GaussianProcessClassifier(kernel=kernel,
	#        random_state=0).fit(x_train, y_train)
	#print(gpc.score(x_train, y_train))

	print("Tree")
	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)
	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)
	print(y_pred)

	'''
	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)
	print("Tree")
	from sklearn.multioutput import ClassifierChain
	clf = ClassifierChain(tree.DecisionTreeClassifier())
	clf = clf.fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)
	'''
	'''
	#print("GradientBoostingClassifier")
	#from sklearn.ensemble import GradientBoostingClassifier
	#clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,
	#    max_depth=10, random_state=0).fit(x_train, y_train)
	#print(clf.score(x_train, y_train))
	#print("R-squared score", clf.score(x_test, y_test))
	#y_pred = clf.predict(x_test)
	#prediction_diff(y_pred, y_test)

	print("RFC")
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf = clf.fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)'''

	'''
	print("ETC")
	from sklearn.ensemble import ExtraTreesClassifier
	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)
	clf = ExtraTreesClassifier(n_estimators=100000, random_state=0)
	clf = clf.fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)
	'''

	print("DTC")
	from sklearn.tree import DecisionTreeClassifier
	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)
	clf2 = DecisionTreeClassifier()
	clf2 = clf2.fit(x_train, y_train)
	print(clf2.score(x_train, y_train))
	print("R-squared score", clf2.score(x_test, y_test))
	y_pred = clf2.predict(x_test)
	prediction_diff(y_pred, y_test)

	from sklearn.ensemble import VotingClassifier
	eclf1 = VotingClassifier(estimators=[
        ('t', clf), ('dtc', clf2)], voting='hard')
	eclf1 = eclf1.fit(x_train, y_train)
	print(eclf1.score(x_train, y_train))
	print("R-squared score", eclf1.score(x_test, y_test))
	y_pred = eclf1.predict(x_test)
	prediction_diff(y_pred, y_test)

	from sklearn.ensemble import ExtraTreesRegressor
	# Create a QuantileTransformer object
	qt = QuantileTransformer(output_distribution='normal').fit(x_train)
	#x_train = qt.fit_transform(x_train)
	#x_test = qt.fit_transform(x_test)
	'''
	print("NN")

	#from sklearn.preprocessing import StandardScaler
	#scaler = StandardScaler()
	#x_train = scaler.fit_transform(x_train) 
	#x_test = scaler.fit_transform(x_test) 
	from sklearn.preprocessing import PowerTransformer

	#pt = PowerTransformer(method='yeo-johnson', standardize=True)
	#x_train = pt.fit_transform(x_train) 
	#x_test = pt.fit_transform(x_test) 

	from sklearn.preprocessing import RobustScaler
	#transformer = RobustScaler().fit(x_train)
	#x_train = transformer.transform(x_train)
	#x_test = transformer.transform(x_test)
	sc = MinMaxScaler().fit(x_train)
	x_train = sc.transform(x_train)

	x_test = sc.transform(x_test)
	#solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000
	from sklearn.neural_network import MLPClassifier
	clf = MLPClassifier(max_iter=1000)
	clf = clf.fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)
'''

	#print("HDBSCAN")
	#from sklearn.cluster import HDBSCAN
	#hdb = HDBSCAN(min_cluster_size=20)
	#hdb.fit(x_train)

	#print("OPTICS")
	#from sklearn.cluster import OPTICS
	#clustering = OPTICS(eps=1000, min_samples=2).fit(x_train)

	#print("Covariance")
	#from sklearn.covariance import EllipticEnvelope
	#cov = EllipticEnvelope(random_state=0).fit(x_train)


	print("Isolation Forest")
	x_train = qt.transform(x_train)
	x_test = qt.transform(x_test)
	from sklearn.ensemble import IsolationForest
	#n_estimators = 34, random_state=0, max_features=5
	clf = IsolationForest(contamination=0.01).fit(x_train)
	y_pred = clf.predict(x_test)
	total = 0
	for x in range(len(y_pred)):
		if y_pred[x] == -1:
			if y_test[x] == "normal":
				total+=1
				print(y_test[x])
	print(total)
	print(np.count_nonzero(y_pred == -1))
	print(np.count_nonzero( y_test == "normal"))
	#print(len(y_pred))
	#print(set(y_pred))
	'''
	sc = MinMaxScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.fit_transform(x_test)
	'''
	'''
	from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
	clf = QuadraticDiscriminantAnalysis()
	clf.fit(x_train, y_train)
	print(clf.score(x_train, y_train))
	print("R-squared score", clf.score(x_test, y_test))
	y_pred = clf.predict(x_test)
	prediction_diff(y_pred, y_test)'''



	# Fit and transform the data
	x_train = qt.transform(x_train)
	x_test = qt.transform(x_test)
	from sklearn.covariance import EllipticEnvelope
	clf = EllipticEnvelope(contamination=0.001, support_fraction=1).fit(x_train)
	y_pred = clf.predict(x_test)
	total=0
	for x in range(len(y_pred)):
		if y_pred[x] == -1:
			if y_test[x] == "normal":
				total+=1
				print(y_test[x])
	print(total)
	print(np.count_nonzero(y_pred == -1))
	print(np.count_nonzero( y_test == "normal"))
	print(len(y_test))

	#predict_diff(y_pred)
	#print("Isolation forest test")
	#y_pred = clf.predict(y_test)
	#predict_diff(y_pred)


	from sklearn.cluster import Birch
	#x_train = qt.fit_transform(x_train)
	#x_test = qt.fit_transform(x_test)
	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	#x_train = scaler.fit_transform(x_train) 
	#x_test = scaler.fit_transform(x_test) 
	from sklearn.preprocessing import PowerTransformer

	#pt = PowerTransformer(method='yeo-johnson', standardize=True)
	#x_train = pt.fit_transform(x_train) 
	#x_test = pt.fit_transform(x_test) 

	#from sklearn.preprocessing import RobustScaler
	#transformer = RobustScaler().fit(x_train)
	#x_train = transformer.transform(x_train)
	#x_test = transformer.transform(x_test)

	brc = Birch(n_clusters=5)
	brc = brc.fit(x_train)
	A = brc.predict(x_test)
	print(np.count_nonzero(A == 0), "0")
	print(np.count_nonzero(A == 1), "1")
	print(np.count_nonzero(A == 2),"2")
	print(np.count_nonzero(A == 3),"3")
	print(np.count_nonzero(A == 4),"4")
	print(np.count_nonzero(y_test == "normal"),"n")
	print(np.count_nonzero(y_test == "u2r"),"u")
	print(np.count_nonzero(y_test == "dos"),"d")
	print(np.count_nonzero(y_test == "r2l"),"r")
	print(np.count_nonzero(y_test == "probe"),"p")
	total = 0
	total2 = 0
	total3 = 0
	total4 = 0
	total5 = 0
	for x in range(len(A)):
		if A[x] == 2 and y_test[x]=="normal":
			total+=1
		if A[x] == 1 and y_test[x]=="u2r":
			total2+=1
		if A[x] == 2 and y_test[x]=="dos":
			total3+=1
		if A[x] == 3 and y_test[x]=="r2l":
			total4+=1
		if A[x] == 4 and y_test[x]=="probe":
			total5+=1
	print(total)
	print(total2)
	print(total3)
	print(total4)
	print(total5)
	'''
	from sklearn.cluster import MiniBatchKMeans
	brc = MiniBatchKMeans(branching_factor, n_clusters=5)
	brc = brc.partial_fit(x_train)
	A = brc.predict(x_test)
	print(np.count_nonzero(A == 0), "0")
	print(np.count_nonzero(A == 1), "1")
	print(np.count_nonzero(A == 2),"2")
	print(np.count_nonzero(A == 3),"3")
	print(np.count_nonzero(A == 4),"4")
	print(np.count_nonzero(y_test == "normal"),"n")
	print(np.count_nonzero(y_test == "u2r"),"u")
	print(np.count_nonzero(y_test == "dos"),"d")
	print(np.count_nonzero(y_test == "r2l"),"r")
	print(np.count_nonzero(y_test == "probe"),"p")


	from sklearn import linear_model

	clf = linear_model.SGDOneClassSVM(random_state=42)
	clf =clf.fit(x_train)
	y_pred = clf.predict(x_test)
	total=0
	for x in range(len(y_pred)):
		if y_pred[x] == -1:
			if y_test[x] == "normal":
				total+=1
				print(y_test[x])
	print(total)
	print(np.count_nonzero(y_pred == -1))
	print(np.count_nonzero( y_test == "normal"))
	'''
	'''
	from sklearn.neighbors import LocalOutlierFactor
	clf = LocalOutlierFactor(n_neighbors=2)
	print(len(clf.fit_predict(x_train)))
	y_pred = clf.predict(x_test)
	print(np.count_nonzero(y_pred == -1))
	'''
	'''
	from sklearn.cluster import DBSCAN
	clustering = DBSCAN(eps=1000, min_samples=100).fit(train)
	print(set(clustering.labels_))'''
	
	'''
	print("OneClass SVM")
	from sklearn.svm import OneClassSVM
	clf = OneClassSVM(gamma='auto').fit(train)
	y_pred = clf.predict(test)
	print(set(y_pred))
	print(np.count_nonzero(y_pred == -1))
	print(np.count_nonzero( y_test == "normal"))
	'''

	'''
	print("Means")
	from sklearn.cluster import KMeans
	clf = KMeans(n_clusters=34, random_state=0, n_init="auto").fit(train)
	y_pred = clf.predict(test)
	print(set(y_pred))
	print(np.count_nonzero(y_pred == 1))
	print(np.count_nonzero( y_test == "normal"))
	'''

	#####
	#sc = MinMaxScaler()
	#x_train = sc.fit_transform(x_train)
	#x_test = sc.fit_transform(x_test)

	from sklearn.naive_bayes import GaussianNB
	print("GaussianNB")
	clf = GaussianNB()

	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print("R-squared score", clf.score(x_train, y_train))
	print("R-squared score test", clf.score(x_test, y_test))
	prediction_diff(y_pred, y_test)



if __name__ == "__main__":
	test()
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