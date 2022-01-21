import pandas as pd
from preprocessing import setup_train_test_idx, train_test_split
# 采样解决数据不平衡问题
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
# 从csv中读取数据
def import_elliptic_data_from_csvs():
    df_classes = pd.read_csv('elliptic_txs_classes.csv')
    df_edges = pd.read_csv('elliptic_txs_edgelist.csv')
    df_features = pd.read_csv('elliptic_txs_features.csv', header=None)
    return df_classes, df_edges, df_features

# 载入数据
def load_elliptic_data(only_labeled=True, drop_node_id=True):
    df_classes, df_edges, df_features = import_elliptic_data_from_csvs()
    df_features = rename_features(df_features)
    df_classes = rename_classes(df_classes)
    df_combined = combine_dataframes(df_classes, df_features, only_labeled)
    if drop_node_id == True:
        X = df_combined.drop(columns=['id', 'class'])
    else:
        X = df_combined.drop(columns='class')
    y = df_combined['class']
    # SMOTE过采样
    #X, y = SMOTE().fit_resample(X, y)
    #随机欠采样
    # rus = RandomUnderSampler(random_state=0)
    # X, y = rus.fit_resample(X, y)
    # smote_enn采样（综合）
    # smote_enn = SMOTEENN(random_state=0)
    # X, y = smote_enn.fit_resample(X, y)
    return X, y



# 重命名标签
def rename_classes(df_classes):
    df_classes.replace({'class': {'1': 1, '2': 0, 'unknown': 2}}, inplace=True)
    return df_classes

# 重命名特征
def rename_features(df_features):
    df_features.columns = ['id', 'time_step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                          range(72)]
    return df_features

# 将数据做左连接
def combine_dataframes(df_classes, df_features, only_labeled=True):
    df_combined = pd.merge(df_features, df_classes, left_on='id', right_on='txId', how='left')
    if only_labeled == True:
        df_combined = df_combined[df_combined['class'] != 2].reset_index(drop=True)
    df_combined.drop(columns=['txId'], inplace=True)
    return df_combined


def calc_occurences_per_timestep():
    X, y = load_elliptic_data()
    X['class'] = y
    occ = X.groupby(['time_step', 'class']).size().to_frame(name='occurences').reset_index()
    return occ


# 划分训练集测试集
def run_elliptic_preprocessing_pipeline(last_train_time_step, last_time_step, only_labeled=True,
                                        drop_node_id=True):
    X, y = load_elliptic_data(only_labeled, drop_node_id)
    train_test_idx = setup_train_test_idx(X, last_train_time_step, last_time_step)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, train_test_idx)

    return X_train_df, X_test_df, y_train, y_test