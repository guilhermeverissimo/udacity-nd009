import matplotlib.pyplot as plt
import numpy as np
import pywt
import time
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_sample(idx, data, lx_cols, ly_cols, rx_cols, ry_cols):
    """Plotting left and right eyes movements for a sample.

    Args:
        idx: Index of observation in data.
        data: Data to be sampled with idx    
        lx_cols: Columns labels for left eye values on X axis
        ly_cols: Columns labels for left eye values on Y axis
        rx_cols: Columns labels for right eye values on X axis
        ry_cols: Columns labels for right eye values on Y axis
    """

    # Plotting left and right eyes movements for samples
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,4))
    
    ax[0].set_title('Eye coordinates on X axis - index:' + str(idx) + ' | subject:' + str(data.loc[idx,['sid']][0]))
    ax[0].plot(np.arange(0, len(lx_cols)), data.loc[idx, lx_cols], label='left', color='b')
    ax[0].plot(np.arange(0, len(rx_cols)), data.loc[idx, rx_cols], label='right', color='r')    
    ax[0].legend(loc='lower left')
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Position')
    ax[0].set_ylim((-1500, 1500))

    ax[1].set_title('Eye coordinates on Y axis  - index:' + str(idx) + ' | subject:' + str(data.loc[idx,['sid']][0]))
    ax[1].plot(np.arange(0, len(ly_cols)), data.loc[idx, ly_cols], label='left', color='b') 
    ax[1].plot(np.arange(0, len(ry_cols)), data.loc[idx, ry_cols], label='right', color='r')
    ax[1].legend(loc='lower left')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Position')
    ax[1].set_ylim((-1500, 1500))

    plt.show()

    
def train_predict(clf,  X_train, y_train, X_test, y_test, comments, logloss=True, train=True): 
    ''' Fit and predict data, evaluating metrics.
    Args:
        clf: Classifier
        X_train: Features training set
        y_train: Income training set
        X_test: Features testing set
        y_test: Income testing set
        comments: Textual comments and notes
        logloss: Wheter to evaluate logloss metric        
        train: If false does not train, just predict
    Returns:
        res: Dict containing results
    '''
    
    
    res = {} # results
    start = time.time()

    if train:
        clf.fit(X_train, y_train)
    
    # predicted values
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    
    # storing results
    res['train_acc'] = accuracy_score(y_train, pred_train)
    res['test_acc'] = accuracy_score(y_test, pred_test)
    
    res['train_f1'] = f1_score(y_train, pred_train, average='weighted')
    res['test_f1'] = f1_score(y_test, pred_test, average='weighted')
    
    
    # predicted prob
    if logloss:
        pred_probs_train = clf.predict_proba(X_train)
        pred_probs_test = clf.predict_proba(X_test)
        # storing results
        res['train_logloss'] = log_loss(y_train, pred_probs_train)
        res['test_logloss'] = log_loss(y_test, pred_probs_test)
    else:
        res['train_logloss'] = None
        res['test_logloss'] = None
    
    # storing results
    res['notes'] = comments
    res['time'] = time.time() - start     
        
    # Return the results
    return res, pred_train, pred_test

def aggPosition(x):
    """Aggregate position data inside a segment

    Args:
        x: Position values in a segment        
    
    Returns:
        Aggregated position (single value)
    """

    return x.mean()

def aggVelocity(x):
    """Aggregate velocity inside a segment of data

    Args:
        x: Position values in a segment        
    
    Returns:
        Aggregated velocity (single value)
    """
    v = np.gradient(x)
    
    # Return the highest absolute peak
    return v[np.argmax(np.abs(v))]


def aggAcceleration(x):
    """Aggregate acceleration inside a segment of data

    Args:
        x: Position values in a segment        
    
    Returns:
        Aggregated velocity (single value)
    """
    a = np.gradient(np.gradient(x))
    
    # Return the highest absolute peak
    return a[np.argmax(np.abs(a))]

def applyfiltering(x, wavelet='db2', mode='smooth'):
    """Applying low-pass filtering for the elimination of high-frequency effects 
    using single level Discrete Wavelet Transform (DWT)

    Args:
        x: Input signal
        wavelet: Wavelet to use
        mode: Signal extension mode
    
    Returns:
        Filtered signal
    """
    # Applying Discrete Wavelet Transform (DWT)
    cA, cD = pywt.dwt(x, wavelet, mode=mode)
#    # Recovering signal filtered
#    x_rec = pywt.idwt(cA, None, wavelet, mode=mode)
    
    return np.copy(cA)

def getArrays(data, sid_col, sx_cols, sy_cols, lx_cols, ly_cols, rx_cols, ry_cols):    
    """Applying low-pass filtering for the elimination of high-frequency effects 
    using single level Discrete Wavelet Transform (DWT)

    Args:
       data: Dataset with samples of the experiment
       sid_col: Columns with subject identifier
       sx_cols: Columns with stimulus point placements on X axis 
       sy_cols: Columns with stimulus point placements on Y axis 
       lx_cols: Columns with left eye gaze points on X axis
       ly_cols: Columns with left eye gaze points on Y axis
       rx_cols: Columns with right eye gaze points on X axis
       ry_cols: Columns with right eye gaze points on Y axis
    
    Returns:
        Filtered signal
    """
    # Converting to array type
    sid = np.asarray(data[sid_col]) # 'subject id' column
    sx = np.asarray(data[sx_cols]) # 'sx0'..'sx2047' columns
    sy = np.asarray(data[sy_cols]) # 'sy0'..'sy2047' columns
    lx = np.asarray(data[lx_cols]) # 'lx0'..'lx2047' columns
    ly = np.asarray(data[ly_cols]) # 'ly0'..'ly2047' columns
    rx = np.asarray(data[rx_cols]) # 'rx0'..'rx2047' columns
    ry = np.asarray(data[ry_cols]) # 'ry0'..'ry2047' columns
    return sid, sx, sy, lx, ly, rx, ry


def doSplit(lx, ly, rx, ry, nsegments, trim=False):
    """" Segments each sample in a number of parts with the same lenght
    Arguments: 
        lx: Left eye gaze points on X axis
        ly: Left eye gaze points on X axis
        rx: Left eye gaze points on X axis
        ry: Left eye gaze points on X axis
        nsegments: Number of segments        
    Returns:
        slx: List of arrays, one array per segment
        sly: List of arrays, one array per segment
        srx: List of arrays, one array per segment
        sry: List of arrays, one array per segment  
    """
    assert len(lx)==len(ly)==len(rx)==len(ry), 'Arrays with different size'
    
    # Splitting and defining segments    
    nrows = len(lx)
    slx = []
    sly = []
    srx = []
    sry = []
    
    for i in np.arange(0,nrows):
        if trim & (nsegments>2):            
            slx.append(np.array_split(lx[i], nsegments)[1:-1]) 
            sly.append(np.array_split(ly[i], nsegments)[1:-1])
            srx.append(np.array_split(rx[i], nsegments)[1:-1])
            sry.append(np.array_split(ry[i], nsegments)[1:-1])
        else:
            slx.append(np.array_split(lx[i], nsegments)) 
            sly.append(np.array_split(ly[i], nsegments))
            srx.append(np.array_split(rx[i], nsegments))
            sry.append(np.array_split(ry[i], nsegments))
            
        
    return slx, sly, srx, sry

def aggFeatures(sl, prefix):
    ##########################################################################
    # Creating additional features
    """" Compute feature aggreation for list of segments
    Arguments: 
        sl: List of arrays
        prefix: Identification of list of arrays
    Returns:
        features: Dataframe with aggregated features
    """
    
    nrows = len(sl)    
    aggfeatures = {} #dict with every new feature
    for i in np.arange(0,nrows):
        nsegments = len(sl[i])
        ifeatures = {}
        for j in np.arange(0,nsegments):
            values = sl[i][j]            
            # Eyes position
            featname = prefix + str(j)        
            featvalue = aggPosition(values)
            ifeatures[featname] = featvalue
            # Eyes velocity
            featname = 'v' + prefix  + str(j)
            featvalue = aggVelocity(values)
            ifeatures[featname] = featvalue
            # Eyes acceleration
            featname = 'a' + prefix + str(j)
            featvalue = aggAcceleration(values)
            ifeatures[featname] = featvalue
        
        aggfeatures[i] = ifeatures

    # Dataframe with aggregated features
    features = pd.DataFrame(aggfeatures).T
    
    return features


# =============================================================================
# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# (modified)
# =============================================================================
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]        

    fig, ax = plt.subplots(figsize=(19,16))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm
