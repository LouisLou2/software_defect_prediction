import os

import arff
import pandas as pd

def arff_to_parquet(folder_path, target_folder):
    # 遍历文件夹下所有ARFF文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.arff'):
            with open(folder_path + file_name, 'r') as file:
                dataset = arff.load(file)
                cols = [attr[0] for attr in dataset['attributes']]
                data = pd.DataFrame(dataset['data'], columns=cols)
                # 将Defective列N转0，Y转1
                if 'Defective' in data.columns:
                    data['Defective'] = data['Defective'].map({'N': 0, 'Y': 1})
                target_filename = target_folder + file_name.split('.')[0] + '.parquet'
                data.to_parquet(target_filename)
                print(f'{file_name} has been converted to {target_filename}')