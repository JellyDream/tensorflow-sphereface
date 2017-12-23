import os
def file_list(data_dir, list_txt):
    filenames = []
    with open(list_txt, 'rt') as f:
        for line in f:
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames

used_count = -1
def load_data(filenames, batch_size, do_shuffle):
    global used_count
    if used_count == -1:
       if do_shuffle:
            random.shuffle(filenames)
       used_count = 0
    if used_count > len(filenames) - BATCH_SIZE:
        used_count = 0
        if do_shuffle:
            random.shuffle(filenames)

    batch_data = np.zeros((batch_size, 112, 96, 3))
    batch_label = np.zeros((batch_size), dtype = np.int32)
    for i in range(batch_size):
        img_name = filenames[used_count + i].split(' ')[0]
        label = int(filenames[used_count + i].split(' ')[1])
        img = cv2.imread(img_name)
        batch_data[i,:] = (img-127.5)/128.0
        batch_label[i] = label
    used_count = used_count + BATCH_SIZE
    return batch_data, batch_label

def get_multistep_lr(iter_):
    return basic_learning_rate * math.pow(factor, sum([1 for value in step_value if value < iter_]))

def get_lambda_value(iter_):
    return lambda_base * math.pow( lambda_gamma * (iter_ -begin_iteration) + 1, -lambda_power)

