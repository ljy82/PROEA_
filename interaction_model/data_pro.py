import numpy as np

def load_data(fr_crossview_file, en_crossview_file, subclass_file, disjoint_file):
    fr_crossview_data = []
    en_crossview_data = []
    subclass_data = []
    disjoint_data = []

    with open(fr_crossview_file, 'r') as f:
        for line in f:
            fr_crossview_data.append(line.strip().split())

    with open(en_crossview_file, 'r') as f:
        for line in f:
            en_crossview_data.append(line.strip().split())

    with open(subclass_file, 'r') as f:
        for line in f:
            cleaned_line = [item.strip('<>') for item in line.strip().split()]
            subclass_data.append(cleaned_line)
            
    with open(disjoint_file, 'r') as f:
        for line in f:
            disjoint_data.append(line.strip().split())

    return fr_crossview_data, en_crossview_data, subclass_data, disjoint_data

def generate_class_to_index(fr_crossview_data, en_crossview_data):
    class_to_index = {}
    index = 0

    # 从 fr_crossview 数据中提取本体
    for crossview in fr_crossview_data:
        if crossview[1] not in class_to_index:
            class_to_index[crossview[1]] = index
            index += 1

    # 从 en_crossview 数据中提取本体
    for crossview in en_crossview_data:
        if crossview[1] not in class_to_index:
            class_to_index[crossview[1]] = index
            index += 1

    return class_to_index

def preprocess_data(subclass_data, disjoint_data, class_to_index):
    # print(subclass_data)
    # print(disjoint_data)
    print("class_to_index",class_to_index)
    subclass_indices = []
    disjoint_indices = []
    
    for subclass in subclass_data:
        if subclass[0] in class_to_index and subclass[2] in class_to_index:
            subclass_indices.append([class_to_index[subclass[0]], class_to_index[subclass[2]]])

    
    for disjoint in disjoint_data:
        # print(disjoint)
        if disjoint[0] in class_to_index and disjoint[2] in class_to_index:
            disjoint_indices.append([class_to_index[disjoint[0]], class_to_index[disjoint[2]]])

    return np.array(subclass_indices), np.array(disjoint_indices)