import pandas as pd
import pyshark
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

#if we decide to use flask
from flask import Flask, jsonify

app = Flask(__name__)

#load the trained model
model = joblib.load('decision_tree_model.joblib')

#define columns in dataset
COLS = [
    'duration', 'protocol_type', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
]

#define mappings for categorical features
PMAP = {'icmp': 0, 'tcp': 1, 'udp': 2}
FMAP = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6, 'S2': 7, 
        'RSTOS0': 8, 'S3': 9, 'OTH': 10}

#attack type mapping
ATTACK_MAP = {'normal': 0, 'attack': 1}

#capture file path
capture_file_path = '/packetfiles/live_capture.pcap'

#init scaler (fit with training data or use saved scaler)
scaler = MinMaxScaler()

def preprocess_packet(packet):
    """Preprocess packet data for model prediction."""
    #extract features from the packet and apply mapping 
    features = {
        'duration': float(packet.tcp.time_relative) if hasattr(packet, 'tcp') else 0,
        'protocol_type': PMAP.get(packet.transport_layer, -1),
        'src_bytes': int(packet_lenth),

        #add other features as extracted from packet attributes...

    }

    #convert features to dataframe format
    features_df = pd.DataFrame([features])

    #scale features using MinMaxScaler 
    features_df = scaler.transform(features_df)

    return features_df

@app.get('/packets', methods=['GET'])
def get_packets():
    """Capture packets, preprocess, predict, and serve as JSON response."""
    capture = pyshark.FileCapture(capture_file_path)
    packet_data = []

    for packet in capture:
        try:
            #preprocess
            features = preprocess_packet(packet)

            #predict
            prediction = model.predict(features)[0]

            #interpret
            status = 'attack' if prediction != ATTACK_MAP['normal'] else 'normal'
            packet_info = {
                'protocol': packet.transport_layer,
                'source_ip': packet.ip.src,
                'dest_ip': packet.ip.dst,
                'status': status
            }
            packet_data.append(packet_info)
        except Exception as e:
            print(f"Error processing packet: {e}")

    return jsonify(packet_data)

if __name__ == '__main__':
    app.run(debug=True)