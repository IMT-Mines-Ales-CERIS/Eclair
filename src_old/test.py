from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def trainModelkFolds_nbc(train_xdata, train_ydata, NB_SPLIT, random_state):
    #split train data to NB_SPLIT blocks
    kf = KFold(n_splits=NB_SPLIT)
    # print(kf.get_n_splits(train_ydata))

    print(kf)

    train_k=[]
    test_k=[]
    for i, (train_index, test_index) in enumerate( kf.split(train_xdata, train_ydata) ):
        train_k.append( train_index  )
        test_k.append( test_index )
    print(train_k)
    print(test_k)
    y_cal=[]
    # for k in range( NB_SPLIT ):
    #     X_train, X_cal, y_train, y_cal_tmp = train_test_split(train_xdata[train_k[k], :], train_ydata[train_k[k]], test_size=0.2, random_state=random_state)


from enum import Enum

# class syntax

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(Color.BLUE)


x = [
    [1,1,1,1,1],
    [2,2,2,2,2],
    [3,3,3,3,3],
    [4,4,4,4,4],
    [1,1,1,1,1],
    [2,2,2,2,2],
    [3,3,3,3,3],
    [4,4,4,4,4],
]
y = [
    1,
    2,
    3,
    4,
    1,
    2,
    3,
    4,
]

trainModelkFolds_nbc(x, y, 4, 42)