'''df'''
print ('practice')
import pandas as pd
import numpy as np
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

credit_raw = pd.read_csv('creditcard.csv')

def simple_read():
    #credit_raw = pd.read_csv('creditcard.csv')
    X = credit_raw.ix[:,0:-1]
    amount = X['Amount']
    amount_int = amount%5
    amount_int_tf = amount_int == 0.00
    X['amount_int_tf'] = amount_int_tf
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = credit_raw["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
    #X_train = X_train[:22000]
    #y_train = y_train[:22000]
    #print (X_train.shape)
    print('Done simple-read')
    return X_train, X_test, y_train, y_test, X, y

def resmapling(size_multipler = 1, scaler_type = 'MinMax'):


    X = credit_raw
    amount = X['Amount']
    amount_int = amount%5
    amount_int_tf = amount_int == 0.00
    X['amount_int_tf'] = amount_int_tf
    #scaler = MinMaxScaler()
    #X = scaler.fit_transform(X)
    #y = credit_raw["Class"]


    if scaler_type == 'MinMax':
        scaler_1 = MinMaxScaler()
    else:
        scaler_1 = StandardScaler()

    number_records_fraud = len(X[X.Class == 1])
    fraud_indices = np.array(X[X.Class == 1].index)
    normal_indices = X[X.Class == 0].index
    random_normal_indices = np.random.choice(normal_indices, size_multipler*number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])
    y_full = X["Class"]
    #print (y_full)
    X_full = X.ix[:, X.columns != 'Class']
    X_full = scaler_1.fit_transform(X_full)
    X_undersample = X_full[under_sample_indices, :]
    #X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
    y_undersample = y_full[under_sample_indices]
    #X
    #X_full =
    #X_full = scaler_1.fit_transform(X_full)
    #y_full = X["Class"]

    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                        , y_undersample
                                                                                                        , test_size=0.3
                                                                                                        ,
                                                                                                        random_state=0)
    print ('Done re-samping')

    return X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample, X_full, y_full

X_train, _, y_train, _, X_test, y_test = resmapling(size_multipler=3)


def logistic():
    #Logistic Regression
    c = [0.001, 0.01, 100, 100000]
    train_acc = []
    test_acc = []
    for item in c:
        lr_est = linear_model.LogisticRegression(C=item)
        lr_est.fit(X_train, y_train)
        trainacc = lr_est.score(X_train, y_train)
        train_acc.append(trainacc)
        testacc = lr_est.score(X_test, y_test)
        test_acc.append(testacc)
        lr_predicted = lr_est.predict(X_test)
        confusion = confusion_matrix(y_test, lr_predicted)
        #y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
        print (confusion)

    plt.plot(c, train_acc, c, test_acc)
    plt.show()

'''KNN'''
def KNN():
    from sklearn.neighbors import KNeighborsClassifier
    n_nb = [5, 30, 50, 100]
    train_acc = []
    test_acc = []
    for k in n_nb:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        trainacc = knn.score(X_train, y_train)
        train_acc.append(trainacc)
        testacc = knn.score(X_test, y_test)
        test_acc.append(testacc)
        knn_predicted = knn.predict(X_test)
        confusion = confusion_matrix(y_test, knn_predicted)
        #y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
        print (confusion)

    plt.plot(n_nb, train_acc, n_nb, test_acc)
    plt.show()

'''Linear SVM'''
def linear_svm():
    from sklearn.svm import SVC
    c_list = [0.0001, 1, 10000]
    train_acc = []
    test_acc = []
    for c in c_list:
        clf = SVC(kernel='linear', C=c).fit(X_train, y_train)
        trainacc = clf.score(X_train, y_train)
        train_acc.append(trainacc)
        testacc = clf.score(X_test, y_test)
        test_acc.append(testacc)
        lr_predicted = clf.predict(X_test)
        confusion = confusion_matrix(y_test, lr_predicted)
        # y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
        print(confusion)

    plt.plot(c, train_acc, c, test_acc)
    plt.show()


'''POLY SVM'''
def poly_svm():
    from sklearn.svm import SVC
    c_list = [10, 1000, 100000, 1000000, 10000000]
    d_list = [5]
    train_acc = []
    test_acc = []
    for d in d_list:
        for c in c_list:
            clf = SVC(kernel='poly', C=c, degree=d, max_iter=500000).fit(X_train, y_train)
            trainacc = clf.score(X_train, y_train)
            train_acc.append(trainacc)
            testacc = clf.score(X_test, y_test)
            test_acc.append(testacc)
            lr_predicted = clf.predict(X_test)
            confusion = confusion_matrix(y_test, lr_predicted)
            confusion_2 = confusion_matrix(y_train, clf.predict(X_train))
            # y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
            print(confusion)
            print (confusion_2)

#plt.plot(c, train_acc, c, test_acc)
#plt.show()


''' rbf kernal svm'''
def rbf_svm():
    from sklearn.svm import SVC
    c_list = [10, 1000, 100000, 1000000, 10000000]
    gamma_list = [0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []
    for g in gamma_list:
        for c in c_list:
            clf = SVC(kernel='rbf', C=c, gamma=g, max_iter=500000).fit(X_train, y_train)
            trainacc = clf.score(X_train, y_train)
            train_acc.append(trainacc)
            testacc = clf.score(X_test, y_test)
            test_acc.append(testacc)
            lr_predicted = clf.predict(X_test)
            confusion = confusion_matrix(y_test, lr_predicted)
            confusion_2 = confusion_matrix(y_train, clf.predict(X_train))
                # y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
            print(confusion)
            print(confusion_2)


'''sigmoid kernal svm'''
def sigmoid_svm():
    from sklearn.svm import SVC
    c_list = [10, 1000, 100000, 1000000, 10000000]
    gamma_list = [0.01, 0.1, 1, 10, 100]
    train_acc = []
    test_acc = []
    for g in gamma_list:
        for c in c_list:
            clf = SVC(kernel='sigmoid', C=c, gamma=g, max_iter=500000).fit(X_train, y_train)
            trainacc = clf.score(X_train, y_train)
            train_acc.append(trainacc)
            testacc = clf.score(X_test, y_test)
            test_acc.append(testacc)
            lr_predicted = clf.predict(X_test)
            confusion = confusion_matrix(y_test, lr_predicted)
                # y_score_lr = lr.fit(X_train, y_train).decision_function(X_test)
            print(confusion)

'''Naive_Bayes'''
def Naive_Bayes():
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    gbn = GaussianNB()
    gbn.fit(X_train, y_train)
    trainacc = gbn.score(X_train, y_train)
    print (trainacc)
    testacc = gbn.score(X_test, y_test)
    print (testacc)
    gbn_pre = gbn.predict(X_test)
    confusion = confusion_matrix(y_test, gbn_pre)
    print (confusion)

def simple_nn():

    def pre(w1, b1, w2, b2, data_set):
        layer1 = tf.nn.relu(tf.matmul(data_set, w1) + b1)
        layer2 = tf.matmul(layer1, w2) + b2
        res = tf.nn.sigmoid(layer2)
        return res

    def nn_structure(num_step=100001,batch_size = 22000,keep_prob=0.8,beta=10):
        graph = tf.Graph()
        with graph.as_default():
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 31))
            tf_train_label = tf.placeholder(tf.float32, shape=(batch_size,1))
            tf_test_dataset = tf.constant(X_test, dtype=tf.float32)
            tf_test_label = tf.constant(y_test)
            weights_layer1 = tf.Variable(tf.truncated_normal([31, 20],stddev=np.sqrt(31)))
            bias_layer1 = tf.Variable(tf.zeros([20]))
            weights_layer2 = tf.Variable(tf.truncated_normal([20, 1],stddev=np.sqrt(20)))
            bias_layer2 = tf.Variable(tf.zeros([1]))

            logits_layer1 = tf.matmul(tf_train_dataset, weights_layer1) + bias_layer1
            layer1_relu = tf.nn.relu(logits_layer1)
            layer1_dropout = tf.nn.dropout(layer1_relu, keep_prob=keep_prob)
            logits_layer2 = tf.matmul(layer1_dropout, weights_layer2) + bias_layer2
            #print (tf_train_label)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_label, logits=logits_layer2)
                + beta*((tf.nn.l2_loss(weights_layer1))+tf.nn.l2_loss(weights_layer2)))

            global_step = tf.Variable(0)
            learningrate = tf.train.exponential_decay(0.5, global_step, 10000, 0.96, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(loss, global_step=global_step)

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
                if step%500==0:
                    print ('Step = ', step)
                    print ('test confusion')
                    print (confusion_test)
                    print ('train confusion')
                    print (confusion_train)

    nn_structure()

def mul_var(percent = 0.6,scaler_type = 'MinMax'):


    X = credit_raw
    amount = X['Amount']
    amount_int = amount%5
    amount_int_tf = amount_int == 0.00
    X['amount_int_tf'] = amount_int_tf
    #scaler = MinMaxScaler()
    #X = scaler.fit_transform(X)
    #y = credit_raw["Class"]


    if scaler_type == 'MinMax':
        scaler_1 = MinMaxScaler()
    else:
        scaler_1 = StandardScaler()

    X = X.sample(frac=1).reset_index(drop=True)

    number_records_fraud = len(X[X.Class == 1])
    number_records_non = len(X[X.Class == 0])
    fraud_indices = np.array(X[X.Class == 1].index)
    normal_indices = X[X.Class == 0].index
    normal_indices_train = normal_indices[:round(number_records_non*percent)]
    normal_indices_cv = normal_indices[round(number_records_non*percent):round(number_records_non*percent)+round((1-percent)/2*number_records_non)]
    normal_indices_test = normal_indices[round(number_records_non*percent)+round((1-percent)/2*number_records_non):]
    normal_indices_train = np.array(normal_indices_train)
    normal_indices_cv = np.array(normal_indices_cv)
    normal_indices_test = np.array(normal_indices_test)

    y = X["Class"]
    #print (y_full)
    X = X.ix[:, X.columns != 'Class']
    X = scaler_1.fit_transform(X)

    X_train_data = X[normal_indices_train, :]
    y_train_data = y[normal_indices_train]
    fraud_indices_up = round(fraud_indices.shape[0]/2)
    fraud_indices_up_ind = np.array(fraud_indices[:fraud_indices_up])
    fraud_indices_down_ind = np.array(fraud_indices[fraud_indices_up:])

    cv_ind = np.concatenate([fraud_indices_up_ind, normal_indices_cv])
    test_ind = np.concatenate([fraud_indices_down_ind, normal_indices_test])

    X_cv_data= X[cv_ind,:]
    y_cv_data = y[cv_ind]
    X_test_data = X[test_ind,:]
    y_test_data = y[test_ind]

    def stats(var_x):
        #print (X.shape)
        X_mean = np.mean(var_x, axis=0)
        #print (X_mean)
        X_cov_matrix = np.cov(var_x.T)
        #print (X_mean)
        #print (X_cov_matrix.shape)
        #X_mean = X_mean[:,]
        return X_mean, X_cov_matrix

    X_train_mean, X_train_cov_matrix = stats(X_train_data)
    print(X_train_mean.shape)
    prob_cv = sp.stats.multivariate_normal.pdf(X_cv_data, X_train_mean, X_train_cov_matrix)
    print (prob_cv.shape)

    fraud_cv_prob = prob_cv[np.nonzero(y_cv_data)]
    normal_cv_prob = prob_cv[np.nonzero(y_cv_data==0)[0]]


    plt.figure()
    plt.hist(fraud_cv_prob, normed=True, bins=20)
    plt.hist(normal_cv_prob, normed=True, bins=20)
    plt.ylabel('Probability')
    plt.show()
    #print (fraud_cv_prob.shape)


mul_var()