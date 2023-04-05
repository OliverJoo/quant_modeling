# cross validation usage
from sklearn.model_selection import (train_test_split, KFold, LeaveOneOut, LeavePOut, ShuffleSplit, TimeSeriesSplit)

data = list(range(1, 11))
print(data)

print("Data: ", train_test_split(data, train_size=.8))

kf = KFold(n_splits=5)
for train, validate in kf.split(data):
    print(" KFold: ", train, validate)

# shuffle for random selection
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train, validate in kf.split(data):
    print(" KFold: ", train, validate)

# leave-out-out method: leave out 1 for validation
# test set overlapped -> leads high correlation btw # of MDL and estimated error
loo = LeaveOneOut()
for train, validate in loo.split(data):
    print(" LeaveOneOut: ", train, validate)

# leave N ea for validation
lpo = LeavePOut(p=2)
for train, validate in lpo.split(data):
    print(" LeavePOut: ", train, validate)

# create Independent sample but overlapped still
ss = ShuffleSplit(n_splits=3, test_size=2, random_state=0)
for train, validate in ss.split(data):
    print(" ShuffleSplit: ", train, validate)

# If sample doesn't IID(independently & identically distributed sample), using TimeSeriesSplit
# aka. Cross-validation method for Finances
tscv = TimeSeriesSplit(n_splits=5)
for train, validate in tscv.split(data):
    print(" TimeSeriesSplit: ", train, validate)
