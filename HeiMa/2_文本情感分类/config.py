"""
配置文件
"""

import pickle


train_batch_size = 512
test_batch_size = 1000

ws = pickle.load(open("./models/ws.pkl", "rb"))

max_len = 80

