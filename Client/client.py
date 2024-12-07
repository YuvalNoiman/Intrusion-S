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
from sklearn.ensemble import IsolationForest
import datetime
import subprocess
from multiprocessing import Process 
import socket
import tkinter

COLS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land' , 'wrong_fragment' , 'urgent' , 'hot','num_failed_logins','logged_in','num_compromised','root_shell' , 'su_attempted' ,'num_root' ,'num_file_creations' ,'num_shells' ,'num_access_files' ,'num_outbound_cmds','is_host_login','is_guest_login' ,'count','srv_count','serror_rate' ,'srv_serror_rate' ,'rerror_rate','srv_rerror_rate' , 'same_srv_rate', 'diff_srv_rate' , 'srv_diff_host_rate' , 'dst_host_count' ,'dst_host_srv_count' ,'dst_host_same_srv_rate','dst_host_diff_srv_rate' ,'dst_host_same_src_port_rate' , 'dst_host_srv_diff_host_rate' , 'dst_host_serror_rate' ,'dst_host_srv_serror_rate' , 'dst_host_rerror_rate' , 'dst_host_srv_rerror_rate' ,'attack_type', 'difficulty']

COLS_KDD99 = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land' , 'wrong_fragment' , 'urgent','count','srv_count','serror_rate' ,'srv_serror_rate' ,'rerror_rate','srv_rerror_rate' , 'same_srv_rate', 'diff_srv_rate' , 'srv_diff_host_rate' , 'dst_host_count' ,'dst_host_srv_count' ,'dst_host_same_srv_rate','dst_host_diff_srv_rate' ,'dst_host_same_src_port_rate' , 'dst_host_srv_diff_host_rate' , 'dst_host_serror_rate' ,'dst_host_srv_serror_rate' , 'dst_host_rerror_rate' , 'dst_host_srv_rerror_rate']

ATTACK_TYPES = {'ftp_write':'r2l', 'normal':'normal', 'rootkit':'u2r', 'imap':'r2l', 'ipsweep':'probe', 'nmap':'probe', 'loadmodule':'u2r', 'multihop':'r2l', 'neptune':'dos', 'teardrop':'dos', 'satan':'probe', 'land':'dos', 'phf':'r2l', 'warezmaster':'r2l', 'smurf':'dos', 'guess_passwd':'r2l', 'buffer_overflow':'u2r', 'perl':'u2r', 'portsweep':'probe', 'spy':'r2l', 'warezclient':'r2l', 'back':'dos', 'pod':'dos','saint':'probe','sqlattack':'u2r','mscan':'probe', 'apache2':'dos','snmpgetattack':'r2l','processtable':'dos','httptunnel':'u2r','ps':'u2r','snmpguess':'r2l','mailbomb':'dos','named':'r2l','sendmail':'r2l','xterm':'u2r','worm':'r2l','xlock':'r2l','xsnoop':'r2l','udpstorm':'dos'}

PMAP = {'icmp':0, 'tcp':1, 'udp':2}
FMAP = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10}

AMAP = {'normal':0, 'r2l':1, 'u2r':2, 'probe':3, 'dos':4}


def ready_data(file_path):
	data = pd.read_csv(file_path, names = COLS)
	#print(data.head())
	#print(len(data.columns))
	#print(data.shape)
	#print(data.isnull().sum())
	#changes to 4 attack categories and normal
	#print(set(data["attack_type"]))
	#print(data["duration"])
	data['attack_type'] = data.attack_type.apply(lambda r:ATTACK_TYPES[r[:]])
	#print(data.head())

	#num_cols = data._get_numeric_data().columns
	#categorical_cols = list(set(data.columns)-set(num_cols))
	#categorical_cols.remove('attack_type')
	#categorical_cols.remove('difficulty')

	#print(categorical_cols)

	# drop rows with NaN
	data = data.dropna(axis='rows')
	
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

def main(mode, alert_type, interval, server_ip, port, client_name):
	train = ready_data("./dataset/KDDTrain+.txt")
	x_train = train.drop(['attack_type','difficulty'], axis = 1)
	#print(x_train.columns)
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
		isf = IsolationForest(contamination=0.01).fit(x_train)
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
		isf = IsolationForest(contamination=0.01).fit(x_train)

	while True:
		run_kdd99_for_interval(interval, "./built_kdd99extractor/kdd99extractor", "features.csv")

		#skip bad lines to prevent errors
		actual_data = pd.read_csv("features.csv", sep=',', header=None, on_bad_lines='skip')
		actual_data.columns = COLS_KDD99
		#print(actual_data.head)
		#print(actual_data.columns)

		for x in [9,10,11,12,13,14,15,16,17,18,19,20,21]:
			actual_data.insert(loc=x, column=str(x), value=0)
		actual_data.insert(loc=41, column="attack", value="normal")
		actual_data.insert(loc=42, column="difficulty", value=10)

		#print(actual_data)
		#print(len(actual_data.columns))
		actual_data.to_csv("features_edited.csv", header=None, index=None, mode='w')
		time.sleep(3)
		#df = pd.read_csv("features_edited.csv", sep=',', header=None)
		#print(df.head())

		actual_data = ready_data("features_edited.csv")
		actual_data = actual_data.drop(['attack_type','difficulty'], axis = 1)
		#print(actual_data.head())
		#print(actual_data.columns)
		if len(actual_data) == 0:
		        continue
		if mode == "1":
			attack_predictions = clf.predict(actual_data)
		elif mode == "2":
			actual_data = sc.fit_transform(actual_data)
			attack_predictions = clf.predict(actual_data)
		elif mode == "3":
			actual_data = qt.fit_transform(actual_data)
			attack_predictions_ee = ee.predict(actual_data)
			print(set(attack_predictions_ee))
			attack_predictions_brc = brc.predict(actual_data)
			print(set(attack_predictions_brc))
			attack_predictions_isf = isf.predict(actual_data)
			print(set(attack_predictions_isf))
			if list(attack_predictions_ee).count(-1) > list(attack_predictions_ee).count(1) and list(attack_predictions_brc).count(1) > list(attack_predictions_brc).count(2) and list(attack_predictions_isf).count(-1) > list(attack_predictions_isf).count(1):
			      attack_predictions == ["attack", "attack"] 
			else:
			      attack_predictions = ["normal"]    
		else:
			attack_predictions = vc.predict(actual_data)
			actual_data = qt.fit_transform(actual_data)
			attack_predictions_ee = ee.predict(actual_data)
			print(set(attack_predictions_ee))
			attack_predictions_brc = brc.predict(actual_data)
			print(set(attack_predictions_brc))
			attack_predictions_isf = isf.predict(actual_data)
			print(set(attack_predictions_isf)) 

		if alert_type == "1":
			machine_side_alert(attack_predictions)
		elif alert_type == "2":
			attack = machine_side_alert(attack_predictions, log=False)
			send_to_server(server_ip, port, client_name, attack)
		else:
			attack = machine_side_alert(attack_predictions)
			send_to_server(server_ip, port, client_name, attack)
		#break
	print("Program ended")
		
def machine_side_alert(attack_predictions, log=True):
	print(len(attack_predictions))
	print(list(attack_predictions).count("normal"))
	attack_prediction_set = set(attack_predictions)
	print(attack_prediction_set)
	if attack_prediction_set != {"normal"}:
		attack = True
		if log==True:	
			make_pop_up_async()
	else:
		attack = False
	if log==True:
		create_log_file(attack)
	return attack

def send_to_server(server_ip, port, client_name, attack):
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
	sock.connect((server_ip, port)) 
	timestamp = datetime.datetime.now()
	sock.send((str(attack)+ "," + client_name + "," + str(timestamp)).encode()) 
	#sock.close()


def make_pop_up():
	pop_up = tkinter.Tk()
	pop_up.geometry("+400+500")
	pop_up["bg"] = "red"
	attack = tkinter.Label(pop_up, text='ALERT ALERT ALERT!!!!!', font=("Arial", 50))
	attack.pack()
	attack["bg"] = "red"
	alert = 'We suspect there to be an attack!!!!!'
	#style = tkinter.ttk.Style()
	#style.configure("alert.TLabel", font=("Arial", 64))
	alert_m = tkinter.Message(pop_up, text=alert)
	alert_m.config(bg='red')
	alert_m.pack()
	alert_m.mainloop()

def make_pop_up_async():
	p = Process(target=make_pop_up, args=())
	p.start()

def create_log_file(attack):
	if attack == True:
		message = "\nThere was an attack\n"
	else:
		message = "\nThere was no attack\n"
	with open("log.txt", "a") as output_file:
		timestamp = datetime.datetime.now()
		output_file.write(str(timestamp))
		output_file.write(message)

def run_kdd99(kdd99_path, output_file_path):
	myoutput = open(output_file_path, 'w')
	#print("file created")
	process = subprocess.call(['sudo', kdd99_path], stdout=myoutput)
	myoutput.close()

def run_kdd99_for_interval(interval, kdd99_path, output_file_path):
	p = Process(target=run_kdd99, args=(kdd99_path, output_file_path,))
	p.start()
	time.sleep(interval)
	p.terminate()
	time.sleep(3)

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

		if alert_type in modes[1:3]:
			server_ip = input("Enter server ip address: ")
			port = int(input("Enter server socket port: "))
			client_name = input("Enter client name with no commas: ")
		else:
			server_ip = "127.0.0.1"
			port = 8080
			client_name = "client"

		if alert_type not in modes[0:3]:
			print("Try again")
		else:
			print("\n")
			break

		print("There are three ways to get alerted: \n\n(1) Machine side alerts \n(2) Server side alerts\n(3) Both machine side and server side alerts\n")
	while True:
		interval = input("Enter the the time in seconds between each alert (has to be a whole number). Recommendation of at least 30! : ")
		if not interval.isnumeric():
			print("Try again")
		else:
			print("\n")
			break
	main(mode, alert_type, int(interval), server_ip, port, client_name)
