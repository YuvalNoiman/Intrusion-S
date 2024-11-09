import pandas as pd
import pyshark
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

capture_file_path = '/packetfiles/live_capture.pcap'
capture = pyshark.FileCapture(capture_file_path)