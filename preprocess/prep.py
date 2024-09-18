import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from data_stat.data_trait import print_data_distribution_binary


def combine_multi_file_to_df(filenames,basedir=''):
    assert len(filenames) > 0, "No file to combine"
    df_list=[]
    data = pd.read_parquet(basedir+'/'+filenames[0])
    df_list.append(data)
    for i in range(1,len(filenames)):
        data=pd.read_parquet(basedir+'/'+filenames[i])
        df_list.append(data)
    data = pd.concat(df_list, ignore_index=True)
    return data


def split_data_with_oversampling(data, test_size=0.1,overs_strategy=0.5):
    X = pd.DataFrame(data.drop(['Defective'], axis=1))
    y = pd.DataFrame(data['Defective'])

    # 分层随机划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,
        test_size=test_size,
        stratify = y,
        random_state=1234
    )
    print_data_distribution_binary("Train", y_train)
    print_data_distribution_binary("Test", y_test)

    sm = SMOTE(random_state=1234, k_neighbors=5,sampling_strategy=overs_strategy)  # for oversampling minority data
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print_data_distribution_binary("Train after oversampling", y_train)

    return X_train, X_test, y_train, y_test