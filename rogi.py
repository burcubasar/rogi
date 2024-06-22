import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os,glob

def load_df(data_path, train=True):
    if train: 
        fname = 'dummy_train.csv'
    else:
        fname = 'dummy_test.csv'
    fpath = glob.glob(os.path.join(data_path,fname))[0]
    df = pd.read_csv(fpath)  
    return df



if __name__ == "__main__":
    path = r'C:\Users\basarb3\OneDrive - Medtronic PLC\Desktop'
    df = load_df(path)
    X = df.loc[:, df.columns != 'status']
    y = df.loc[:, df.columns == 'status']
    df = load_df(path,train=False) 
    Xtest = df.loc[:, df.columns != 'status']
    ytest = df.loc[:, df.columns == 'status']
    
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=0)
    logistic_regression= LogisticRegression()

    logistic_regression.fit(X_train,y_train)
    y_pred=logistic_regression.predict(X_test)
    print(y_pred)
    print(y_test.to_numpy().ravel())