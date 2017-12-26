import os
import random
import numpy as np
import cv2
import math
data_root_dir = 'data/'
data_list = 'txt/webface_list.txt'

BATCH_SIZE = 100
M_VALUE = 4

basic_learning_rate = 0.1
weight_decay = 0.0005
step_value = [16000, 20000, 24000]
max_iter = 28000
factor = 0.1

lambda_base = 1000
lambda_gamma = 0.12
lambda_power = 1
lambda_min = 5
begin_iteration = 0

