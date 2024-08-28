import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import sklearn
if sklearn.__version__ <= "0.17.1":
    print('The scikit-learn version is smaller or equal {}.'.format(sklearn.__version__))
    from sklearn import cross_validation #test_train_split
else:
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    from sklearn.model_selection import train_test_split, cross_val_score
import scipy as sp #for encoding #takes most common class
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xgb

pd.options.display.float_format = '{:.1f}'.format
#fix random seed for reproducibility
seed = 4711
np.random.seed(seed)
rand_val = np.random.randint(0,999999)
# Prepare feature matrix and target vector
X = []
y = []
# Setup classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

def extract_features(x, y, z):
    features = []
    features.append(np.mean(x))
    features.append(np.mean(y))
    features.append(np.mean(z))
    features.append(np.std(x))
    features.append(np.std(y))
    features.append(np.std(z))
    features.append(np.min(x))
    features.append(np.min(y))
    features.append(np.min(z))
    features.append(np.max(x))
    features.append(np.max(y))
    features.append(np.max(z))
    return np.array(features)

def AccuracyScore(y_true,y_pred):
    label = ["Running", "Sitting", "Standing", "Walking"]
    TP=0
    FP=0
    i=0
    while i < y_true.size:
        if(y_true[i] == y_pred[i]):
            TP+=1
        else:
            FP+=1 #0 Running, 1 Sitting, 2 Standing, 3 Walking
            print("Predicted: ", label[y_pred[i]])
            print("Actual: ", label[y_true[i]])
        i += 1
    return TP / y_true.size

def to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan

def process_csv(fname):
    column_names = ['Activity',
                    'ID',
                    'Time',
                    'ax',
                    'ay',
                    'az']

    if(os.path.isfile(fname)==True):
        df = pd.DataFrame()
        df_item = pd.read_csv(fname, header=None, names=column_names, skipinitialspace=True)
        df_item = df_item.truncate(before=150, after=(df_item.shape[0]-150)) #trunc 150 parts of noise data
        df = df_item
    else:
        allFiles = [f for f in os.listdir(fname) if os.path.isfile(os.path.join(fname, f))]
        df = pd.DataFrame()
        for fi in allFiles:
            df_item = pd.read_csv(fname + fi, header=None, names=column_names, skipinitialspace=True)
            df_item = df_item.truncate(before=150, after=(df_item.shape[0]-150)) #trunc 150 parts of noise data
            df = df.append(df_item, ignore_index=True)

    #print dataframe info
    print('num rows df: ', (df.shape[0]))
    print('num columns df: ', df.shape[1])
    #convert to float
    df['ID'] = df['ID'].apply(to_float)
    df['ax'] = df['ax'].apply(to_float)
    df['ay'] = df['ay'].apply(to_float)
    df['az'] = df['az'].apply(to_float)
    #drop all nan
    df.dropna(axis=0, how='any', inplace=True)
    #start label encoding of target classes
    le = LabelEncoder()
    #add a new label column for the label vector
    df['EncActiCol'] = le.fit_transform(df['Activity'].values.ravel())
    #create the actual data
    N_FEATURES = 3 #x,y,z
    segments = []
    labels = []
    #overwrite encoder with non auto class values
    y_out = df['ax'].values
    act = df['Activity'].values
    for k in range(0, len(act), 1):
        if act[k]=="Running":
            y_out[k]=0
        if act[k]=="Sitting":
            y_out[k]=1
        if act[k]=="Standing":
            y_out[k]=2
        if act[k]=="Walking":
            y_out[k]=3
    df['EncActiCol'] = y_out
    print("Total data length: ", len(df))
    for i in range(0, len(df) - 100, 100):
        xs = df['ax'].values[i: i + 100]
        ys = df['ay'].values[i: i + 100]
        zs = df['az'].values[i: i + 100]
        #get most common class in segment
        label = sp.stats.mode(df['EncActiCol'][i: i + 100])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)
    #take the time slice sliced x,y,z arrays and reshape
    X = segments
    y = np.asarray(labels)
    print("Num classes detected: ", le.classes_.size)
    return X, y

#time period sets the time the application will need to detect classes
#e.g. TIME_PERIOD 25 ->  25 * 1/10Hz = 2,5 sec until correct estimation
TIME_PERIOD = 25 #i.e. sliding window size
#the step size sets the amount of overlapping of segments
STEP_SIZE = 5
X_batch, y_batch = process_csv('./data/') #training data dir 
for r in range(0, len(X_batch) - 1, 1):
    for i in range(0, len(X_batch[r][1]) - TIME_PERIOD, STEP_SIZE):
        x_segment_arr = X_batch[r][0][i:i + TIME_PERIOD]
        y_segment_arr = X_batch[r][1][i:i + TIME_PERIOD]
        z_segment_arr = X_batch[r][2][i:i + TIME_PERIOD]
        x_segment = []
        y_segment = []
        z_segment = []
        for k in range(TIME_PERIOD):
            x_segment.append(x_segment_arr[k])
            y_segment.append(y_segment_arr[k])
            z_segment.append(z_segment_arr[k])        
        features = extract_features(np.array(x_segment), np.array(y_segment), np.array(z_segment))
        X.append(features)
        y.append(int(y_batch[r]))  #label is the label of the middle of the segment

# convert features and labels to numpy array 
X = np.array(X)
y = np.array(y)
#split most of the data to a traning set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_val)

# Then, split the 80% training data into 90% training + 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=rand_val)

# X_train, y_train: 72% of the data
# X_val, y_val: 8% of the data
# X_test, y_test: 20% of the data
#print shape
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)
print('x_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('x_val shape: ', X_val.shape)
print('y_val shape: ', y_val.shape)

# fit the training data
model.fit(X_train, y_train)

# Predict on the test set and evaluate
y_pred_test = model.predict(X_test)
accuracy_test = AccuracyScore(y_test, y_pred_test)
print(metrics.classification_report(y_test,y_pred_test))
print(metrics.confusion_matrix(y_test,y_pred_test))
print(f"Test Data Model Accuracy: {accuracy_test:.2f}")
print("--------------------------")

# Predict on the validation set and evaluate
y_pred_val = model.predict(X_val)
accuracy_val = AccuracyScore(y_val, y_pred_val)
print(metrics.classification_report(y_val,y_pred_val))
print(metrics.confusion_matrix(y_val,y_pred_val))
print(f"Validation Data Model Accuracy: {accuracy_val:.2f}")
print("--------------------------------")

