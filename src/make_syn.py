import os

from sklearn.model_selection import KFold, train_test_split
import pandas as pd

from . import config

df_syn = pd.read_csv("input/syn_all.csv")

if __name__ == "__main__":
    X, y = df_syn.f_name, df_syn.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=config.RAND_STATE, shuffle=True)
    df_train = pd.DataFrame({"filename": X_train, "target": y_train}).reset_index(drop=True)
    df_test = pd.DataFrame({"filename": X_test, "target": y_test})
    
    # Add CV indexes to train set
    df_train["kfold"] = -1
    kf = KFold(n_splits=config.N_FOLDS)
    #print(X_train)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_train, y=y_train)):
        df_train.loc[val_idx, 'kfold'] = fold

    df_train.to_csv("input/syn_train.csv", index=False)
    df_test.to_csv("input/syn_test.csv", index=False)
