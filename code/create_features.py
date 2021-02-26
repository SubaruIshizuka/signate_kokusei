import pandas as pd
import numpy as np
import sys,os
import csv
import yaml
from pathlib import Path
import warnings
from util import Logger

import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

sys.path.append(os.pardir)
sys.path.append('../..')
sys.path.append('../../..')
warnings.filterwarnings("ignore")


CONFIG_FILE = '../configs/config.yaml'
with open(CONFIG_FILE, encoding="utf-8") as file:
    yml = yaml.load(file)
RAW_DATA_DIR_NAME = yml['SETTING']['RAW_DATA_DIR_NAME']  # RAWデータ格納場所
FEATURE_DIR_NAME = yml['SETTING']['FEATURE_DIR_NAME']  # 生成した特徴量の出力場所
MODEL_DIR_NAME = yml['SETTING']['MODEL_DIR_NAME'] # モデルの格納場所
REMOVE_COLS = yml['SETTING']['REMOVE_COLS']


#### preprocessing関数を定義 ##########################################################

def get_bins(all_df):
    """binning
    """
    # bin_edges = [-1, 25, 45, np.inf]
    # all_df["bin_age"] = pd.cut(all_df["age"], bins=bin_edges, labels=["young", "middle", "senior"]).astype("object")
    bin_edges = [-1, 20, 30, 40, 50, 60, np.inf]
    all_df["bin_general"] = pd.cut(all_df["age"], bins=bin_edges, labels=["10's", "20's", "30's", "40's", "50's", "60's"]).astype("object")
    return all_df

def get_cross_cate_features(all_df):
    """カテゴリ変数×カテゴリ変数
    """
    obj_cols = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship', 
        'race', 
        'sex',
        'bin_general'
    ]
    for cols in itertools.combinations(obj_cols, 2):
        all_df["{}_{}".format(cols[0], cols[1])] = all_df[cols[0]] + "_" + all_df[cols[1]]
    return all_df

def get_cross_num_features(all_df):
    """数値変数×数値変数
    """
    all_df["prod_age_educationnum"] = all_df["age"] * all_df["education-num"]
    all_df["ratio_age_educationnum"] = all_df["age"] / all_df["education-num"]
    return all_df

def get_agg_features(all_df):
    """集約特徴量
    """
    cate_cols = [
        'workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'bin_general',
       'workclass_education', 'workclass_marital-status',
       'workclass_occupation', 'workclass_relationship', 'workclass_race',
       'workclass_sex', 'workclass_bin_general', 'education_marital-status',
       'education_occupation', 'education_relationship', 'education_race',
       'education_sex', 'education_bin_general', 'marital-status_occupation',
       'marital-status_relationship', 'marital-status_race',
       'marital-status_sex', 'marital-status_bin_general',
       'occupation_relationship', 'occupation_race', 'occupation_sex',
       'occupation_bin_general', 'relationship_race', 'relationship_sex',
       'relationship_bin_general', 'race_sex', 'race_bin_general',
       'sex_bin_general'
    ]
    group_values = [
        "age",
        "education-num",
        "prod_age_educationnum",
        "ratio_age_educationnum"
    ]
    for col in cate_cols:
        for group in group_values:
#             all_df.loc[:, "mean_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).mean()[group]) # 平均
            all_df.loc[:, "std_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).std()[group])   # 分散
#             all_df.loc[:, "max_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).max()[group])   # 最大値
            # all_df.loc[:, "min_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).min()[group])   # 最小値
            # all_df.loc[:, "nunique_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).nunique()[group])   # uniaue
            # all_df.loc[:, "median_{}_{}".format(col, group)] = all_df[col].map(all_df.groupby(col).median()[group])   # 中央値
    return all_df

def get_relative_features(all_df):
    """相対値
    """
    cate_cols = [
        'workclass', 'education', 'marital-status', 'occupation',
       'relationship', 'race', 'sex', 'native-country', 'bin_general',
       'workclass_education', 'workclass_marital-status',
       'workclass_occupation', 'workclass_relationship', 'workclass_race',
       'workclass_sex', 'workclass_bin_general', 'education_marital-status',
       'education_occupation', 'education_relationship', 'education_race',
       'education_sex', 'education_bin_general', 'marital-status_occupation',
       'marital-status_relationship', 'marital-status_race',
       'marital-status_sex', 'marital-status_bin_general',
       'occupation_relationship', 'occupation_race', 'occupation_sex',
       'occupation_bin_general', 'relationship_race', 'relationship_sex',
       'relationship_bin_general', 'race_sex', 'race_bin_general',
       'sex_bin_general'
    ]
    group_values = [
        "age",
        "education-num",
        "prod_age_educationnum",
        "ratio_age_educationnum"
    ]
    # カテゴリごとの平均との差
    for col in cate_cols:
        for group in group_values:
            df = all_df.copy()
            df.loc[:, "mean_{}_{}".format(col, group)] = df[col].map(all_df.groupby(col).mean()[group]) # 平均
            all_df.loc[:, "{}_diff_{}".format(col, group)] = df[group] - df["mean_{}_{}".format(col, group)]
    return all_df

def get_freq_features(all_df):
    """frequency encoding
    """
    cate_cols = [
#         "workclass",
#         "education",
#         "marital-status",
        "occupation",
#         "relationship",
#         "race",
#         "sex",
#         "native-country",
        'education_marital-status',
        'marital-status_occupation',
        'education_occupation',
        'marital-status_relationship',
        'education_relationship',
        'marital-status_bin_general',
    ]
    for col in cate_cols:
        freq = all_df[col].value_counts()
        # カテゴリの出現回数で置換
        all_df.loc[:, "freq_{}".format(col)] = all_df[col].map(freq)
    return all_df

def get_labelencoding(all_df):
    """ラベルエンコーディング
    """
    cols = all_df.dtypes[(all_df.dtypes=="object") | (all_df.dtypes=="category")].index
    for col in cols:
        le = LabelEncoder()
        all_df.loc[:, col] = le.fit_transform(all_df[col])
    return all_df


def get_svd(all_df):
    cols = [
        'marital-status_relationship_diff_prod_age_educationnum',
        'workclass_sex',
        'occupation_relationship_diff_age',
        'occupation_sex_diff_ratio_age_educationnum',
        'race_bin_general_diff_prod_age_educationnum',
        'education_bin_general_diff_age',
        'marital-status_sex',
        'relationship_race',
        'education_race',
        'workclass_sex_diff_prod_age_educationnum',
        'marital-status_occupation_diff_prod_age_educationnum',
        'education_occupation_diff_age',
        'workclass_relationship_diff_age',
        'education_marital-status_diff_prod_age_educationnum',
        'relationship_diff_education-num',
        'relationship_diff_age',
        'std_education_occupation_ratio_age_educationnum',
        'marital-status_occupation_diff_age',
        'workclass_relationship_diff_ratio_age_educationnum',
        'std_occupation_bin_general_ratio_age_educationnum',
        'relationship_bin_general_diff_prod_age_educationnum',
        'education_occupation_diff_ratio_age_educationnum',
        'sex_bin_general_diff_prod_age_educationnum',
        'workclass',
        'marital-status_occupation_diff_ratio_age_educationnum',
        'occupation_race_diff_ratio_age_educationnum',
        'std_occupation_race_ratio_age_educationnum',
        'education_race_diff_prod_age_educationnum',
        'workclass_education_diff_ratio_age_educationnum',
        'workclass_marital-status_diff_age',
        'workclass_sex_diff_ratio_age_educationnum',
        'relationship_bin_general_diff_ratio_age_educationnum',
        'workclass_education_diff_prod_age_educationnum',
        'occupation_diff_prod_age_educationnum',
        'occupation',
        'std_workclass_education_ratio_age_educationnum',
        'std_occupation_relationship_ratio_age_educationnum',
        'marital-status_bin_general_diff_ratio_age_educationnum',
        'workclass_education_diff_age',
        'marital-status_bin_general_diff_prod_age_educationnum',
        'occupation_relationship_diff_ratio_age_educationnum',
        'std_education_occupation_age',
        'workclass_marital-status_diff_ratio_age_educationnum',
        'race_sex',
        'std_workclass_occupation_age',
        'std_workclass_race_ratio_age_educationnum',
        'sex_bin_general_diff_age',
        'std_occupation_relationship_age',
        'std_occupation_bin_general_education-num',
        'occupation_bin_general_diff_education-num',
        'occupation_race_diff_prod_age_educationnum',
        'sex_diff_prod_age_educationnum',
        'workclass_sex_diff_age',
        'prod_age_educationnum',
        'occupation_sex_diff_age',
        'sex',
        'std_occupation_bin_general_age',
        'workclass_education_diff_education-num',
        'workclass_diff_prod_age_educationnum',
        'workclass_race_diff_age',
        'race_diff_age',
        'education_relationship_diff_prod_age_educationnum',
        'occupation_sex_diff_prod_age_educationnum',
        'std_relationship_sex_education-num',
        'relationship',
        'race_bin_general_diff_age',
        'std_education_marital-status_age',
        'bin_general',
        'marital-status',
        'occupation_diff_ratio_age_educationnum',
        'race_bin_general_diff_ratio_age_educationnum',
        'education',
        'race_sex_diff_age',
        'sex_bin_general_diff_ratio_age_educationnum',
        'std_workclass_education_education-num',
        'std_workclass_education_age',
        'relationship_race_diff_age',
        'race',
        'marital-status_race',
        'workclass_race_diff_ratio_age_educationnum',
        'std_occupation_race_education-num',
        'std_education_bin_general_age',
        'age',
        'std_occupation_bin_general_prod_age_educationnum',
        'std_education_bin_general_ratio_age_educationnum',
        'occupation_relationship_diff_education-num',
        'workclass_diff_age',
        'education_relationship_diff_ratio_age_educationnum',
        'std_workclass_relationship_ratio_age_educationnum',
        'std_education_bin_general_education-num',
        'marital-status_sex_diff_prod_age_educationnum',
        'marital-status_relationship_diff_ratio_age_educationnum',
        'education_diff_prod_age_educationnum',
        'workclass_diff_ratio_age_educationnum',
        'occupation_race_diff_education-num',
        'std_workclass_bin_general_ratio_age_educationnum',
        'occupation_race_diff_age',
        'std_occupation_relationship_prod_age_educationnum',
        'std_occupation_relationship_education-num',
        'workclass_bin_general_diff_education-num',
        'std_workclass_education_prod_age_educationnum',
        'occupation_diff_age',
        'std_relationship_sex_age',
        'std_education_race_age',
        'native-country',
        'marital-status_occupation_diff_education-num',
        'std_workclass_bin_general_education-num',
        'std_workclass_race_age',
        'sex_diff_age',
        'std_workclass_relationship_education-num',
        'std_marital-status_relationship_education-num',
        'workclass_marital-status_diff_education-num',
        'workclass_relationship_diff_education-num',
        'marital-status_sex_diff_ratio_age_educationnum',
        'relationship_race_diff_ratio_age_educationnum',
        'education_marital-status_diff_age',
        'relationship_race_diff_prod_age_educationnum',
        'race_sex_diff_prod_age_educationnum',
        'workclass_sex_diff_education-num',
        'relationship_diff_prod_age_educationnum',
        'std_relationship_bin_general_education-num',
        'std_workclass_sex_education-num',
        'freq_education_relationship',
        'education_sex_diff_prod_age_educationnum',
        'education_diff_ratio_age_educationnum',
        'race_diff_prod_age_educationnum',
        'std_workclass_marital-status_education-num',
        'std_race_bin_general_education-num',
        'std_workclass_relationship_age',
        'education_sex_diff_ratio_age_educationnum',
        'education_marital-status_diff_ratio_age_educationnum',
        'std_marital-status_occupation_age',
        'marital-status_diff_ratio_age_educationnum',
        'std_workclass_bin_general_prod_age_educationnum',
        'std_education_sex_education-num',
        'std_occupation_sex_age',
        'relationship_sex_diff_age',
        'std_occupation_sex_prod_age_educationnum',
        'freq_education_marital-status',
        'std_workclass_race_education-num',
        'std_marital-status_bin_general_education-num',
        'education_race_diff_ratio_age_educationnum',
        'std_education_bin_general_prod_age_educationnum',
        'relationship_sex_diff_ratio_age_educationnum',
        'marital-status_race_diff_prod_age_educationnum',
        'std_marital-status_occupation_education-num',
        'std_education_marital-status_prod_age_educationnum',
        'marital-status_diff_prod_age_educationnum',
        'race_sex_diff_ratio_age_educationnum',
        'std_occupation_sex_ratio_age_educationnum',
        'std_education_relationship_ratio_age_educationnum',
        'education_relationship_diff_age',
        'std_marital-status_occupation_prod_age_educationnum',
        'marital-status_race_diff_age',
        'relationship_diff_ratio_age_educationnum',
        'std_education_relationship_age',
        'std_marital-status_occupation_ratio_age_educationnum',
        'occupation_sex_diff_education-num',
        'education_sex_diff_age',
        'std_occupation_race_age',
        'education_race_diff_education-num',
        'std_workclass_education-num',
        'marital-status_diff_age',
        'std_occupation_race_prod_age_educationnum',
        'marital-status_diff_education-num',
        'sex_diff_ratio_age_educationnum',
        'relationship_bin_general_diff_education-num',
        'std_education_relationship_prod_age_educationnum',
        'std_education_relationship_education-num',
        'education_marital-status_diff_education-num',
        'std_workclass_sex_ratio_age_educationnum',
        'std_workclass_bin_general_age',
        'education_relationship_diff_education-num',
        'education_diff_education-num',
        'std_occupation_sex_education-num',
        'std_workclass_marital-status_ratio_age_educationnum',
        'marital-status_race_diff_ratio_age_educationnum',
        'marital-status_relationship_diff_education-num',
        'relationship_race_diff_education-num',
        'race_sex_diff_education-num',
        'occupation_diff_education-num',
        'std_education_race_education-num',
        'relationship_sex_diff_education-num',
        'std_relationship_bin_general_age',
        'std_education_sex_ratio_age_educationnum',
        'std_occupation_ratio_age_educationnum',
        'std_relationship_sex_prod_age_educationnum',
        'education_race_diff_age',
        'marital-status_race_diff_education-num',
        'std_education_marital-status_education-num',
        'education_diff_age',
        'std_education_education-num',
        'marital-status_sex_diff_age',
        'race_bin_general_diff_education-num',
        'std_workclass_relationship_prod_age_educationnum',
        'ratio_age_educationnum',
        'std_workclass_marital-status_age',
        'relationship_sex_diff_prod_age_educationnum',
        'std_occupation_education-num',
        'std_sex_bin_general_education-num',
        'std_relationship_bin_general_ratio_age_educationnum',
        'marital-status_bin_general_diff_education-num',
        'std_relationship_race_ratio_age_educationnum',
        'std_education_race_ratio_age_educationnum',
        'race_diff_ratio_age_educationnum',
        'native-country_diff_prod_age_educationnum',
        'education_sex_diff_education-num',
        'std_education_marital-status_ratio_age_educationnum',
        'std_workclass_marital-status_prod_age_educationnum',
        'marital-status_sex_diff_education-num',
        'std_workclass_age',
        'std_workclass_sex_age',
        'std_education_race_prod_age_educationnum',
        'std_workclass_race_prod_age_educationnum',
        'std_relationship_race_prod_age_educationnum',
        'freq_marital-status_relationship',
        'std_marital-status_sex_prod_age_educationnum',
        'std_workclass_ratio_age_educationnum',
        'std_relationship_race_age',
        'std_workclass_sex_prod_age_educationnum',
        'std_relationship_race_education-num',
        'std_education_sex_age',
        'std_marital-status_race_prod_age_educationnum',
        'std_race_bin_general_ratio_age_educationnum',
        'std_race_sex_ratio_age_educationnum',
        'std_occupation_age',
        'std_sex_bin_general_age',
        'sex_bin_general_diff_education-num',
        'std_marital-status_bin_general_ratio_age_educationnum',
        'freq_marital-status_bin_general',
        'std_marital-status_sex_education-num',
        'std_marital-status_race_age',
        'bin_general_diff_education-num',
        'native-country_diff_ratio_age_educationnum',
        'std_relationship_age',
        'std_marital-status_sex_age',
        'std_education_ratio_age_educationnum',
        'std_marital-status_sex_ratio_age_educationnum',
        'education-num',
        'std_relationship_bin_general_prod_age_educationnum',
        'std_marital-status_bin_general_age',
        'std_marital-status_relationship_ratio_age_educationnum',
        'std_race_bin_general_age',
        'std_relationship_sex_ratio_age_educationnum',
        'sex_diff_education-num',
        'std_sex_bin_general_ratio_age_educationnum',
        'std_bin_general_age',
        'std_workclass_prod_age_educationnum',
        'std_sex_bin_general_prod_age_educationnum',
        'std_race_sex_age',
        'std_education_age',
        'std_marital-status_bin_general_prod_age_educationnum',
        'std_education_sex_prod_age_educationnum',
        'std_marital-status_race_ratio_age_educationnum',
        'std_race_sex_education-num',
        'race_diff_education-num',
        'std_occupation_prod_age_educationnum',
        'std_race_ratio_age_educationnum',
        'std_marital-status_race_education-num',
        'std_race_age',
        'std_race_bin_general_prod_age_educationnum',
        'std_marital-status_age',
        'std_marital-status_education-num',
        'std_relationship_education-num',
        'std_race_sex_prod_age_educationnum',
        'education_occupation_diff_prod_age_educationnum',
        'workclass_bin_general_diff_ratio_age_educationnum',
        'std_workclass_occupation_ratio_age_educationnum',
        'workclass_occupation_diff_education-num',
        'bin_general_diff_prod_age_educationnum',
        'workclass_bin_general_diff_prod_age_educationnum',
        'occupation_bin_general_diff_prod_age_educationnum',
        'std_education_occupation_education-num',
        'workclass_occupation',
        'workclass_race_diff_prod_age_educationnum',
        'marital-status_bin_general_diff_age',
        'std_workclass_occupation_prod_age_educationnum',
        'workclass_marital-status_diff_prod_age_educationnum',
        'occupation_bin_general_diff_ratio_age_educationnum',
        'occupation_sex',
        'std_marital-status_relationship_prod_age_educationnum',
        'education_bin_general_diff_ratio_age_educationnum',
        'std_education_occupation_prod_age_educationnum',
        'occupation_relationship_diff_prod_age_educationnum',
        'workclass_race_diff_education-num',

    ]
    df = all_df.copy().loc[:, cols]
    
    df = get_labelencoding(df)
    
    # 標準化
    for col in df.columns:
        scaler = StandardScaler()
        df.loc[:, col] = scaler.fit_transform(df[col].values.reshape(-1,1))
        df = df.fillna(df[col].mean())

    # svd
    n_components = 5
    svd = TruncatedSVD(n_components=n_components, random_state=35)
    svd.fit(df)
    svd_df = pd.DataFrame(svd.transform(df),
                          columns=["svd_{}".format(i) for i in range(n_components)])
    all_df = pd.concat([all_df, svd_df], axis=1)

    return all_df


##### main関数を定義 ###########################################################################
def main():
    
    # データの読み込み
    train = pd.read_csv(RAW_DATA_DIR_NAME + 'train.csv')
    test = pd.read_csv(RAW_DATA_DIR_NAME + 'test.csv')
    df = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    
    # preprocessingの実行
    df = get_bins(df)
    df = get_cross_cate_features(df)
    df = get_cross_num_features(df)
    df = get_agg_features(df)
    df = get_relative_features(df)
    df = get_freq_features(df)
    df = get_svd(df)
    # df = get_labelencoding(df)
    
    # trainとtestに分割
    train = df.iloc[:len(train), :]
    test = df.iloc[len(train):, :]

    print("train shape: ", train.shape)
    print("test shape: ", test.shape)
    
    # pickleファイルとして保存
    train.to_pickle(FEATURE_DIR_NAME + 'train.pkl')
    test.to_pickle(FEATURE_DIR_NAME + 'test.pkl')
#     logger.info(f'train shape: {train.shape}, test shape, {test.shape}')
    
    # 生成した特徴量のリスト
    features_list = list(df.drop(columns=REMOVE_COLS).columns)  # 学習に不要なカラムは除外
    
    # 特徴量リストの保存
    # features_list = sorted(features_list)
    with open(FEATURE_DIR_NAME + 'features_list.txt', 'wt') as f:
        for i in range(len(features_list)):
            f.write('\'' + str(features_list[i]) + '\',\n')
    
    return 'main() Done!'

    
if __name__ == '__main__':
    
#     global logger
#     logger = Logger(MODEL_DIR_NAME + "create_features" + "/")

    main()
