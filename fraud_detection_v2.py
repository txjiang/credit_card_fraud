import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, \
    roc_curve, recall_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold, cross_val_score
from scipy.stats import multivariate_normal
import os
import time
start_time = time.time()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

credit_raw = pd.read_csv('creditcard.csv')

def pre_pressing_data(n_components = 0.99, plot = False):

    '''Normalization, Adding one Feature, PCA'''
    #credit_raw['amount_int_tf'] = amount_int_tf
    class_into = credit_raw['Class']
    credit_temp = credit_raw.drop(['Class', 'Time'], axis=1)
    #credit_temp['Time'] = transfer_time
    #credit_temp['Amount'] = transfer_amount
    credit_temp = MinMaxScaler().fit_transform(credit_temp)
    pca = PCA(n_components=n_components)
    pca.fit(credit_temp)
    credit_temp = pca.transform(credit_temp)
    df_pre_temp = pd.DataFrame(data=credit_temp)
    #df_pre_temp['amount_int_tf'] = amount_int_tf
    df_pre_temp['Class'] = class_into
    '''Statistical Analysis'''
    df_pre_temp.drop([1, 2, 3], axis=1, inplace=True)
    if plot == True:
        v_features = df_pre_temp.ix[:,:-1].columns
        for i, cn in enumerate(df_pre_temp[v_features]):
            temp1 = df_pre_temp[cn][df_pre_temp.Class == 1]
            #print (temp1.shape)
            temp2 = df_pre_temp[cn][df_pre_temp.Class == 0]
            plt.hist(temp1, bins=50, alpha=0.5, normed=True, label='Fraud')
            #print(temp2.shape)
            plt.hist(temp2, bins=50, alpha=0.5, normed=True, label='Non-Fraud')
            plt.legend(fontsize=30, loc='upper right')
            plt.title('Fraud and Non-Fraud Histogram Comparison for Feature V' + str(i+1), fontsize=40)
            plt.show()
    print (df_pre_temp.shape)
    #print (df_pre_temp.head())
    print ('Date Processed')
    return df_pre_temp

preprocess_data = pre_pressing_data()

def tsne_visual():
    from sklearn.manifold import TSNE
    df = preprocess_data[preprocess_data.Class == 1]
    df = pd.concat([df, preprocess_data[preprocess_data.Class==0].sample(n=10000)], axis=0)
    scalar = MinMaxScaler()
    X_full = scalar.fit_transform(df)
    y = df.ix[:,-1].values
    print (y)
    tsne = TSNE(n_components=2, random_state=0)
    x_test_2d = tsne.fit_transform(X_full)
    color_map = {0: 'yellow', 1: 'blue'}
    plt.figure()
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x_test_2d[y == cl, 0],
                    y=x_test_2d[y == cl, 1],
                    c=color_map[idx],
                    label=cl)
    plt.xlabel('X in t-SNE')
    plt.ylabel('Y in t-SNE')
    plt.legend(loc='upper left')
    plt.title('t-SNE visualization of processed data', fontsize=40)
    plt.show()

tsne_visual()


def reample_and_split(resample = False, size_multipler=1, split_ratio = 0.5):
    if resample == True:
        number_records_fraud = len(preprocess_data[preprocess_data.Class == 1])
        fraud_indices = np.array(preprocess_data[preprocess_data.Class == 1].index)
        normal_indices = preprocess_data[preprocess_data.Class == 0].index
        random_normal_indices = np.random.choice(normal_indices, size_multipler * number_records_fraud, replace=False)
        random_normal_indices = np.array(random_normal_indices)
        under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
        y_full = preprocess_data["Class"]
        X_full = preprocess_data.ix[:, preprocess_data.columns != 'Class']
        scalar = MinMaxScaler()
        X_full = scalar.fit_transform(X_full)

        X_undersample = X_full[under_sample_indices, :]
        # X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
        y_undersample = y_full[under_sample_indices]

        X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(
                                                                                                X_undersample,
                                                                                                y_undersample,
                                                                                                test_size=split_ratio,
                                                                                                random_state=0)
        print('Done re-samping')
        return X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample, X_full, y_full

    else:
        X = preprocess_data.ix[:, 0:-1]
        y = credit_raw["Class"]

        scalar = MinMaxScaler()
        X = scalar.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=1)
        return X_train, X_test, y_train, y_test

# Resample with 10% of fraud
re_X_train, _, re_y_train, _, re_X_test, re_y_test = reample_and_split(resample=True, size_multipler=9)
print (sum(re_y_train==1))
print (re_y_train.shape)
# Regular
no_X_train, no_X_test, no_y_train, no_y_test = reample_and_split()

'''ML'''

def logistic(resample = False):
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    c = [0.001, 0.01, 1, 10, 100]
    recall_list = []
    for item in c:
        lr_est = linear_model.LogisticRegression(C=item, penalty='l1')
        lr_est.fit(X_train, y_train)
        lr_predicted = lr_est.predict(X_train)
        print ('C-parameter: ' + str(item))
        print ('Training Recall :', recall_score(y_train, lr_predicted))
        recall_list.append(recall_score(y_train, lr_predicted))
        confusion = confusion_matrix(y_train, lr_predicted)
        print ('Training Confusion: ')
        print (confusion)
        print('Test Report')
        y_pred = lr_est.predict(X_test)
        print(classification_report(y_test, y_pred))
    best_c = c[recall_list.index(max(recall_list))]
    print ('Best C: ', best_c)
    lr = linear_model.LogisticRegression(C=best_c, penalty='l1')
    lr.fit(X_train, y_train)
    score = lr.decision_function(X_test)
    y_pred = lr.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print ('Test Set Confusion: ')
    print (con_matrix)
    print ('Area under Curve: ', roc_auc_score(y_test, y_pred))
    fpr, tpr, thr = roc_curve(y_test, score)
    roc_auc = auc (fpr, tpr)
    plt.figure()
    plt.title('ROC Chart for Logistic Regression', fontsize = 50)
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.ylabel('True Positive Rate', fontsize = 50)
    plt.xlabel('False Positive Rate', fontsize = 50)
    plt.show()
    print('Test Report')
    print(classification_report(y_test, y_pred))

#logistic(resample=True)
'''KNN'''
def KNN(resample = False):
    from sklearn.neighbors import KNeighborsClassifier
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    n_nb = [5, 10, 25, 50]
    recall_list = []
    for item in n_nb:
        print ('number-of-neighbour-parameter: ' + str(item))
        knn = KNeighborsClassifier(n_neighbors=item)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_train)
        print('Training Recall :', recall_score(y_train, y_pred))
        recall_list.append (recall_score(y_train, y_pred))
        confusion = confusion_matrix(y_train, y_pred)
        print('Training Confusion: ')
        print(confusion)
    best_n_nb = n_nb[recall_list.index(max(recall_list))]
    print('Best C: ', best_n_nb)
    knn = KNeighborsClassifier(n_neighbors=best_n_nb)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print('Test Set Confusion: ')
    print (con_matrix)
    print ('Area under Curve: ', roc_auc_score(y_test, y_pred))
    print('Test Report')
    print(classification_report(y_test, y_pred))

#KNN(resample=True)

'''Linear SVM'''
def linear_svm(resample = False):
    from sklearn.svm import SVC
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    c_list = [0.01, 0.1, 1, 10, 100, 1000]
    recall_list = []
    for item in c_list:
        SVM = SVC(kernel='linear', C=item, max_iter=100000)
        SVM.fit(X_train, y_train)
        SVM_predicted = SVM.predict(X_train)
        print ('C-parameter: ' + str(item))
        print ('Training Recall :', recall_score(y_train, SVM_predicted))
        recall_list.append(recall_score(y_train, SVM_predicted))
        confusion = confusion_matrix(y_train, SVM_predicted)
        print ('Training Confusion: ')
        print (confusion)
    best_c = c_list[recall_list.index(max(recall_list))]
    print ('Best C: ', best_c)
    SVM = SVC(kernel='linear', C=best_c, max_iter=100000)
    SVM.fit(X_train, y_train)
    score = SVM.decision_function(X_test)
    y_pred = SVM.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print ('Test Set Confusion: ')
    print (con_matrix)
    fpr, tpr, thr = roc_curve(y_test, score)
    roc_auc = auc (fpr, tpr)
    plt.figure()
    plt.title('ROC Chart for Logistic Regression', fontsize = 50)
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.ylabel('True Positive Rate', fontsize = 50)
    plt.xlabel('False Positive Rate', fontsize = 50)
    plt.show()
    print('Area under Curve: ', roc_auc_score(y_test, y_pred))
    print('Test Report')
    print(classification_report(y_test, y_pred))
#linear_svm(resample = True)

'''Poly SVM'''
def poly_svm(resample = False):
    from sklearn.svm import SVC
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    c_list = [0.01, 0.1, 1, 10, 100, 1000]
    recall_list = []
    for item in c_list:
        SVM = SVC(kernel='poly', C=item, degree=5, max_iter=500000)
        SVM.fit(X_train, y_train)
        SVM_predicted = SVM.predict(X_train)
        print ('C-parameter: ' + str(item))
        print ('Training Recall :', recall_score(y_train, SVM_predicted))
        recall_list.append(recall_score(y_train, SVM_predicted))
        confusion = confusion_matrix(y_train, SVM_predicted)
        print ('Training Confusion: ')
        print (confusion)
    best_c = c_list[recall_list.index(max(recall_list))]
    print ('Best C: ', best_c)
    SVM = SVC(kernel='poly', C=best_c, degree=5, max_iter=500000)
    SVM.fit(X_train, y_train)
    score = SVM.decision_function(X_test)
    y_pred = SVM.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print ('Test Set Confusion: ')
    print (con_matrix)
    print('Area under Curve: ', roc_auc_score(y_test, y_pred))
    fpr, tpr, thr = roc_curve(y_test, score)
    roc_auc = auc (fpr, tpr)
    plt.figure()
    plt.title('ROC Chart for Poly-Kernalized Logistic Regression', fontsize = 50)
    plt.plot(fpr, tpr, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.ylabel('True Positive Rate', fontsize = 50)
    plt.xlabel('False Positive Rate', fontsize = 50)
    plt.show()
    print('Test Report')
    print(classification_report(y_test, y_pred))

#poly_svm(resample=True)
''' rbf kernal svm'''
def rbf_svm(resample = False):
    from sklearn.svm import SVC
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    c_list = [0.01, 0.1, 1, 10, 100, 1000]
    gamma_list = [0.01, 0.1, 1, 10, 100]
    recall_list_g = []
    g_c = []
    for g in gamma_list:
        recall_list_c = []
        for c in c_list:
            SVM = SVC(kernel='rbf', C=c, gamma=g, max_iter=500000)
            SVM.fit(X_train, y_train)
            SVM_predicted = SVM.predict(X_train)
            print ('=========================================')
            print('Gamma-parameter: ' + str(g))
            print('c-parameter: ' + str(c))
            print('Training Recall :', recall_score(y_train, SVM_predicted))
            recall_list_c.append(recall_score(y_train, SVM_predicted))
            confusion = confusion_matrix(y_train, SVM_predicted)
            print('Training Confusion: ')
            print(confusion)
            y_pred = SVM.predict(X_test)
            con_matrix = confusion_matrix(y_test, y_pred)
            print('Test Confusion: ')
            print(con_matrix)
            print ('Test Report')
            print (classification_report(y_test, y_pred))
        recall_list_g.append(max(recall_list_c))
        g_c.append((g, c_list[recall_list_c.index(max(recall_list_c))]))
    best_g_c = g_c[recall_list_g.index(max(recall_list_g))]
    best_g = best_g_c[0]
    best_c = best_g_c[1]
    print('Best C: ', best_c)
    print('Best g: ', best_g)
    SVM = SVC(kernel='rbf', C=best_c, gamma=best_g, max_iter=500000)
    SVM.fit(X_train, y_train)
    score = SVM.decision_function(X_test)
    y_pred = SVM.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print('Test Set Confusion: ')
    print(con_matrix)
    print('Area under Curve: ', roc_auc_score(y_test, y_pred))
    fpr, tpr, thr = roc_curve(y_test, score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.title('ROC Chart for RBF-Kernalized Logistic Regression', fontsize=50)
    plt.plot(fpr, tpr, 'b')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate', fontsize=50)
    plt.xlabel('False Positive Rate', fontsize=50)
    plt.show()
    print('Test Report')
    print(classification_report(y_test, y_pred))

#rbf_svm(resample=True)

'''Multi-Variant Detection'''
def multi_gus_detection(resample = False):
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    #X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.7, random_state=0)
    X_mean = np.mean(X_train, axis=0)
    X_variance = np.std(X_train, axis=0)
    X_cov = np.cov(X_train.T)
    prob = multivariate_normal.pdf(X_train, X_mean, X_cov)
    def select_threhold():
        bestEps = 0
        bestF1 = 0
        F1 = 0
        min_prob = np.min(prob)
        max_prob = np.max(prob)
        stepsize = (max_prob-min_prob)/1000
        for item in np.arange(min_prob, max_prob, stepsize):
            tp = 0
            fp = 0
            fn = 0

            for i,j in zip(prob, y_train):
                if i < item and j == 1:
                    tp = tp + 1
                elif i < item and j == 0:
                    fp = fp + 1
                elif i >= item and j == 1:
                    fn = fn + 1

            if (tp+fp) == 0 or (tp+fn) == 0:
                continue
            else:
                prec = tp/(tp+fp)
                rec = tp/(tp+fn)
                F1 = 2*prec*rec/(prec+rec)
                print ('recall:', rec)
                print ('precision: ', prec)


            if F1 > bestF1:
                bestF1 = F1
                bestEps = item

        return bestF1, bestEps

    bestF1, bestEps = select_threhold()
    print (bestEps, bestF1)
    prob_test = multivariate_normal.pdf(X_test, np.mean(X_test, axis=0), np.cov(X_test.T))
    res = prob_test < bestEps
    confusion_test = confusion_matrix(y_test, np.array(res))
    print(confusion_test)
    print('Area under Curve: ', roc_auc_score(y_test, np.array(res)))
    print('Test Report')
    print(classification_report(y_test, np.array(res)))

#multi_gus_detection(resample=True)

def Naive_Bays(resample = False):
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    gbn = GaussianNB()
    gbn.fit(X_train, y_train)
    y_pred = gbn.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print('Test Set Confusion: ')
    print(con_matrix)
    print('Test Report: ')
    print(classification_report(y_test, y_pred))
    print('Area under Curve: ', roc_auc_score(y_test, y_pred))
    print('Test Report')
    print(classification_report(y_test, y_pred))

#Naive_Bays(resample=True)
def random_forest(resample = False):
    from sklearn.ensemble import RandomForestClassifier
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test
    n_est_list = [10, 50, 100, 500]
    max_list = [5, 10, 15, None]
    recall_list_max = []
    max_est = []
    for g in n_est_list:
        recall_list_max_temp = []
        for c in max_list:
            clf = RandomForestClassifier(n_estimators=g, max_depth=c, random_state=0)
            clf.fit(X_train, y_train)
            clf_predicted = clf.predict(X_train)
            print('=========================================')
            print('Number of estimator: ' + str(g))
            print('Max Depth: ' + str(c))
            print('Training Recall :', recall_score(y_train, clf_predicted))
            recall_list_max_temp.append(recall_score(y_train, clf_predicted))
            confusion = confusion_matrix(y_train, clf_predicted)
            print('Training Confusion: ')
            print(confusion)
            y_pred = clf.predict(X_test)
            con_matrix = confusion_matrix(y_test, y_pred)
            print('Test Confusion: ')
            print(con_matrix)
            print('Test Report')
            print(classification_report(y_test, y_pred))
        recall_list_max.append(max(recall_list_max_temp))
        max_est.append((g, max_list[recall_list_max_temp.index(max(recall_list_max_temp))]))
    #print (recall_list_max)
    #print (max_est)
    best_g_c = max_est[recall_list_max.index(max(recall_list_max))]
    best_g = best_g_c[0]
    best_c = best_g_c[1]
    print('Best Number of Estimator: ', best_g)
    print('Best Max Depth: ', best_c)
    clf = RandomForestClassifier(n_estimators=best_g, max_depth=best_c, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    con_matrix = confusion_matrix(y_test, y_pred)
    print('Test Set Confusion: ')
    print(con_matrix)
    print('Area under Curve: ', roc_auc_score(y_test, y_pred))
    print('Test Report')
    print(classification_report(y_test, y_pred))
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print (indices)

    plt.figure()
    plt.title("Feature Importance Based on Random Forest", fontsize=40)
    plt.bar(range(X_train.shape[1]), importances[indices], alpha=0.5)
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
#random_forest(resample = False)

def simple_nn(resample = False):
    if resample == False:
        X_train = no_X_train
        y_train = no_y_train
        X_test = no_X_test
        y_test = no_y_test
    if resample == True:
        X_train = re_X_train
        y_train = re_y_train
        X_test = re_X_test
        y_test = re_y_test

    def pre(w1, b1, w2, b2, data_set):
        layer1 = tf.nn.relu(tf.matmul(data_set, w1) + b1)
        layer2 = tf.matmul(layer1, w2) + b2
        res = tf.nn.sigmoid(layer2)
        return res

    def nn_structure(num_step=1000,batch_size = y_train.shape[0], keep_prob=1):
        graph = tf.Graph()
        with graph.as_default():
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 31))
            tf_train_label = tf.placeholder(tf.float32, shape=(batch_size,1))
            tf_test_dataset = tf.constant(X_test, dtype=tf.float32)
            tf_test_label = tf.constant(y_test)
            weights_layer1 = tf.Variable(tf.truncated_normal([19, 10],stddev=0.15))
            bias_layer1 = tf.Variable(tf.zeros([10]))
            weights_layer2 = tf.Variable(tf.truncated_normal([10, 1],stddev=0.15))
            bias_layer2 = tf.Variable(tf.zeros([1]))

            logits_layer1 = tf.matmul(tf_train_dataset, weights_layer1) + bias_layer1
            layer1_relu = tf.nn.relu(logits_layer1)
            layer1_dropout = tf.nn.dropout(layer1_relu, keep_prob=keep_prob)
            logits_layer2 = tf.matmul(layer1_dropout, weights_layer2) + bias_layer2
            #print (tf_train_label)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_label, logits=logits_layer2))

            optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

            train_prediction = pre(weights_layer1, bias_layer1, weights_layer2, bias_layer2, tf_train_dataset)
            test_prediction = pre(weights_layer1, bias_layer1, weights_layer2, bias_layer2, tf_test_dataset)

        with tf.Session(graph=graph) as session:
            print ('-----Initialized------')
            tf.global_variables_initializer().run()
            for step in range(num_step):
                #offset = (step*batch_size)%(y_train.shape[0]-batch_size)
                batch_data = X_train
                #[offset:(offset+batch_size),:]
                #print (batch_data.shape)
                batch_label = np.reshape(y_train, (batch_size,1))
                #print (batch_label)
                #print(batch_label.shape)
                feed_dict = {tf_train_dataset:batch_data,tf_train_label:batch_label}
                _,l,train_pre, test_pre = session.run([optimizer,loss,train_prediction,test_prediction],feed_dict=feed_dict)
                ######covert prob to answer
                train_pre = train_pre > 0.5
                test_pre = test_pre > 0.5
                confusion_train = confusion_matrix(batch_label, train_pre)
                confusion_test = confusion_matrix(y_test, test_pre)
                if step%50==0:
                    print ('Step = ', step)
                    print ('test confusion')
                    print (confusion_test)
                    print ('train confusion')
                    print (confusion_train)

    nn_structure()

print("--- %s seconds ---" % (time.time() - start_time))