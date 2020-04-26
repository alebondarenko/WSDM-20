"""
Author: Rami Aly, E-mail: `rami.aly@outlook.com`
"""

from keras.callbacks import ModelCheckpoint
import operator
from data_helpers import load_data, extract_hierarchies, remove_genres_not_level, ml, UNSEEN_STRING
import numpy as np
import string
import math
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
import tensorflow as tf
import itertools
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import itertools
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
import traceback
import keras.backend as K
from keras.callbacks import Callback, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing import sequence
import sys
import argparse
import codecs
import json
import os
import scipy
from networks import create_model_cnn, create_model_lstm, create_model_capsule, create_character_model_cnn
import pickle
from sklearn.model_selection import KFold
import pandas as pd
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.sequence import pad_sequences

#All necessary program arguments are stored here
args = None
#the dataset and vocabulary is stored here
data = {}


def mean_confidence_interval(data, confidence=0.95):
    """
    Calculates mean confidence interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def save_scores(results):
    """
    Stores the scores into a file
    """
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../checkpoints','results_' +  args.filename + '.txt')
    out_file = open(filename, 'w')
    metrices = ['f1', 'recall', 'precision', 'accuracy']
    out_file.write("Results" + '\n')
    for i in range(len(results[0])):
        mean, lower_confidence, upper_confidence = mean_confidence_interval([element[i] for element in results])
        print("%s: %0.2f \pm %0.2f"%( metrices[i],(mean*100), ((upper_confidence - mean))))
        out_file.write(metrices[i] + ": " + str(mean*100) + " \pm " + str(((upper_confidence - mean))))
        out_file.write('\n')
    print('\n')
    out_file.close()


def save_predictions(data_test, output):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    '../checkpoints','predictions_split_' + str(args.num_split) + "_"  + args.filename + '.txt')
    out_file = open(filename, 'w')
    if "BINARY" not in args.lang:
        for label in ml.classes_:
            out_file.write(label + '\t')
    else:
        out_file.write("Comparative_Confidence" + '\t')
    out_file.write("question\n")
    vocab_inv = {}
    for key,value in data['vocabulary'].items():
        vocab_inv[value] = key
    for i, question in enumerate(data_test):
        for element in output[i]:
            out_file.write("%0.20f\t"% element)
        question = [vocab_inv[question_ind] for question_ind in question if vocab_inv[question_ind] != "0.0"]
        question = ' '.join(question)
        out_file.write(str(question) + '\n')
    out_file.close()



class Metrics_eval(Callback):
    """
    Callback to receive score after each epoch of training
    """
    def __init__(self,validation_data):
        self.val_data = validation_data

    def eval_metrics(self):
        #dont use term validation_data, name is reserved
        val_data = self.val_data
        X_test = val_data[0]
        y_test = val_data[1]
        output = self.model.predict(X_test, batch_size = args.batch_size)
        for pred_i in output:
            pred_i[pred_i >=args.activation_th] = 1
            pred_i[pred_i < args.activation_th] = 0

        try:
            if len(y_test[0]) <= 1:
                print("Using binary scoring")
                score_mode = 'binary'
            else:
                print("Using micro scoring")
                score_mode = 'micro'
            if 'BINARY' not in args.lang:
                print(classification_report(y_test, output, target_names = ml.classes_))

            return [f1_score(y_test, output, average=score_mode),
            recall_score(y_test, output, average=score_mode),precision_score(y_test, output, average=score_mode)]
        except:
            return [0., 0., 0., 0., 0.]

    def on_epoch_end(self, epoch, logs={}):
        f1, recall, precision = self.eval_metrics()
        print("For epoch %d the scores are F1: %0.4f, Recall: %0.2f, Precision: %0.2f"%(epoch, f1, recall, precision))
        print("\n ###################################################################################################################################################\n")
        # print((str(precision) + '\n' +  str(recall) + '\n' +
        #          str(f1_macro) + '\n' + str(f1) + '\n' + str(acc)).replace(".", ","))



def create_machine_plots(m_type):
    """
    Plots the neural network with keras vis. capabilities
    """
    model = create_model(preload = True)
    plot_model(model, to_file=os.path.join(os.path.dirname(__file__),
     'model_' + args.classifier + 'cnn.png'), show_shapes=True)



def train(model, X_train = None, y_train = None, X_test = None, y_test = None, save = True, early_stopping = True, validation = True):
    """
    Trains a neural network, can use early_stopping and validationsets
    """

    print("Traning Model...")
    callbacks_list = []

    #class_weight = {0:1, 1:0.5}

    lr_decay = LearningRateScheduler(schedule=lambda epoch: args.learning_rate * (args.learning_decay ** epoch))
    callbacks_list.append(lr_decay)
    if early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=0, mode='auto')
        callbacks_list.append(early_stopping)
    if validation:
        metrics_callback = Metrics_eval(validation_data = (data['X_dev'], data['y_dev']))
        callbacks_list.append(metrics_callback)
        model.fit(data['X_train'], data['y_train'], batch_size=args.batch_size, epochs=args.epochs,
         verbose=1, callbacks=callbacks_list, validation_data=(data['X_dev'], data['y_dev'])) # starts training
    else:
        #ADJUSTED FOR CV, CHANGE BACK AFTER DONE (REMOVE VALIDATION DATA HERE)
        metrics_callback = Metrics_eval(validation_data = (X_test, y_test))
        callbacks_list.append(metrics_callback)
        model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=1,
         callbacks = callbacks_list, validation_data = (X_test, y_test))


    if save:
        print("Saving current Model")
        model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
             '../checkpoints', args.filename) + '.h5')


def test(model, data_l, label, do_analysis = False):
    """
    Tests a neural network on the given data
    """
    global data
    print("Testing Model...")
    print(len(data_l))
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../resources', args.filename + '.output')
    if args.mode == 'evaluate' and os.path.exists(results_path):
        print("Loading model output...")
        output_file = open(results_path, 'rb')
        _,_,_,output,_ = pickle.load(output_file)
    else:
        output = model.predict(data_l, batch_size = args.batch_size)
    binary_output = np.array(output, copy = True)
    #print(binary_output)
    for pred_i in output:
        pred_i[pred_i >=args.activation_th] = 1
        pred_i[pred_i < args.activation_th] = 0

    if args.adjust_hierarchy != 'None' and args.adjust_hierarchy != "threshold":
        output = adjust_hierarchy(output_b = output, language = args.lang,
         mode = args.adjust_hierarchy, max_h = args.level)
    elif args.adjust_hierarchy == "threshold":
        output = adjust_hierarchy_threshold(output = output, output_b = binary_output,
         language = args.lang, max_h = args.level, threshold = args.correction_th)

    if args.store_output:
        save_predictions(data_l, binary_output)
    results = {}
    if(len(output[0])) == 1:
        f1 = f1_score(label, output, average='binary')
        recall = recall_score(label, output, average='binary')
        precision =  precision_score(label, output, average='binary')
        accuracy = accuracy_score(label, output)
        results['micro avg'] = {'precision': precision,  'recall': recall, 'f1-score':f1}
        print((str(precision) + '\n' + str(recall) + '\n' + str(f1) + '\n' + str(accuracy)).replace(".", ","))
    else:
        eval_mode = 'micro'
        report = classification_report(label,output, output_dict = True, target_names = ml.classes_)
        for label in report.keys():
            if label in ml.classes_ or label == "micro avg":
                results[label] = report[label]
        print(results)

    if do_analysis:
        output_file = open(results_path, 'wb')
        pickle.dump([data_l, label, output, binary_output, args], output_file)

    return results

def model_cnn(preload = True):
    """
    Creates CNN or loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        print ("Loading model...")
        model = load_model(filepath)
        model.summary()
        return model
    else:
        return create_model_cnn(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang,len(data['y_train'][0]), args.use_static,
         args.init_layer, data['vocabulary'], args.learning_rate, args.num_split)


def model_character_cnn(preload = True):
    """
    Creates CNN or loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        print ("Loading model...")
        model = load_model(filepath)
        model.summary()
        return model
    else:
        return create_character_model_cnn(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang,len(data['y_train'][0]), args.use_static,
         args.init_layer, data['vocabulary'], args.learning_rate)


def model_lstm(preload = True):
    """
    Creates LSTM or loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        print ("Loading model...")
        model = load_model(filepath)
        model.summary()
        return model
    else:
        return create_model_lstm(preload, args.embed_dim, args.sequence_length,
         args.lstm_units, args.lang, len(data['y_train'][0]),
          args.use_static, args.init_layer, data['vocabulary'], args.learning_rate, args.num_split)


def model_capsule(preload = True):
    """
    Creates capsule networkor loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
         '../checkpoints', args.filename + '.h5')
    print(filepath)
    if os.path.isfile(filepath) and preload:
        #model = load_trained_model(filepath, inputs, output)
        print ("Loading model...")
        model = create_model_capsule(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang, len(data['y_train'][0]),
          args.use_static, args.init_layer, data['vocabulary'], args.learning_rate,
          args.dense_capsule_dim, args.n_channels, 3, args.num_split)
        model.load_weights(filepath)
        model.summary()
        return model
    else:
        return create_model_capsule(preload, args.embed_dim, args.sequence_length,
         args.num_filters, args.lang, len(data['y_train'][0]),
          args.use_static, args.init_layer, data['vocabulary'], args.learning_rate,
          args.dense_capsule_dim, args.n_channels, 3, args.num_split)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield np.asarray(l[i:i + n])


def gridsearch():
    #Calculates all combinations of hyperparameters and stores them in List of Lists with same order
    ##cpasule
    arg1 = [8, 16, 32]

    ##lstm + cnn + char_cnn
    #arg1 = [20, 50, 100, 200, 500]

    #arg1 = [25, 50, 100, 200] #200,500, 700 #500
    lrate= [0.001, 0.0005, 0.0001] #0.001
    use_static = [True, False]
    correction_th = [0.5]
    params = [lrate, use_static, correction_th]
    arg1 = [[element] for element in arg1]
    values = arg1
    for param in params:
        values_new = []
        for param_v in param:
            for value in values:
                new_value = value + [param_v]
                values_new.append(new_value)
        values = values_new
    best_result = 0
    best_values = []
    data_voc_original = data['vocabulary'].copy()
    for params in values:
        args.lstm_units = params[0]
        args.num_filters = params[0]
        args.n_channels = params[0]
        args.learning_rate = params[1]
        args.use_static = params[2]
        args.activation_th = params[3]
        results_dict = train_test_cv(data_voc_original)
        results = results_dict['micro avg'][0]
        if results > best_result:
            print("New best result", results)
            best_result = results
            best_values = [params[0], params[1], params[2], params[3]]
    return [best_values, best_result]


def train_test_cv(data_voc_original):
    global data
    X = np.asarray(data['X_train'].tolist() + data['X_test'].tolist())
    y = np.asarray(data['y_train'].tolist() + data['y_test'].tolist())
    n_splits = 10
    kf = KFold(n_splits = n_splits)

    avg_results = {}
    split = 0
    #for num_split, combi in enumerate(combinations):
    for train_index, test_index in kf.split(X):
        train_x, test_x = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        activation = args.activation_th
        args.num_split  = split
        if args.classifier !='character_cnn':
            split+=1
            vocabulary_set = set([])
            data['vocabulary'] = data_voc_original
            print("Calculating Training vocabulary...")
            for text in train_x:
                for id in text:
                    vocabulary_set.add(id)
            print("Old voc size", len(data['vocabulary']))
            print("Set new global training vocabulary")
            new_vocabulary_train = {}
            count = 0
            for entry in data['vocabulary']:
                if data['vocabulary'][entry] in vocabulary_set:
                    new_vocabulary_train[entry] = count
                    count+=1
            new_vocabulary_train[UNSEEN_STRING] = count
            data['vocabulary'] = new_vocabulary_train
            test_x_new = []
            for text in test_x:
                new_text = []
                for id in text:
                    if id not in vocabulary_set:
                        new_text.append(data['vocabulary'][UNSEEN_STRING])
                    else:
                        word = data['vocabulary_inv'][id]
                        new_text.append(data['vocabulary'][word])
                test_x_new.append(new_text)
            test_x = test_x_new
            test_x = np.asarray(test_x)

            train_x_new = []
            for text in train_x:
                new_text = []
                for id in text:
                    word = data['vocabulary_inv'][id]
                    new_text.append(data['vocabulary'][word])
                train_x_new.append(new_text)
            train_x = train_x_new
            train_x = np.asarray(train_x)
        model = create_model(preload = False)
        train(model, X_train = train_x, y_train = train_y, X_test = test_x,
        y_test = test_y, early_stopping = args.use_early_stop, validation = False, save = False)
        results = test(model, data_l = test_x, label = test_y)
        for key, result in results.items():
            if key in avg_results:
                avg_results[key][1] += result['recall']
            else:
                avg_results[key] = [0.,result['recall'],0.]
            avg_results[key][0] +=  result['precision']
            avg_results[key][2] += result['f1-score']
        K.clear_session()
    for key, result in avg_results.items():
        avg_results[key][0]/= n_splits
        avg_results[key][1]/= n_splits
        avg_results[key][2]/= n_splits

    return avg_results


def main():
    """
    Parses input parameters for networks
    """
    global args
    parser = argparse.ArgumentParser(description="CNN for blurbs")
    parser.add_argument('--mode', type=str, default='train', choices=['gridsearch', 'train', 'train_n_models', 'evaluate', 'outlier', 'plot'], help="Mode of the system.")
    parser.add_argument('--classifier', type=str, default='cnn', choices=['cnn','lstm', 'capsule', 'character_cnn'], help="Classifier architecture of the system.")
    parser.add_argument('--lang', type=str, default='COMPQ_BINARY_YAN', help="Which dataset to use")
    parser.add_argument('--dense_capsule_dim', type=int, default=16, help = 'Capsule dim of dense layer')
    parser.add_argument('--n_channels', type=int, default=50, help = 'number channels of primary capsules')
    parser.add_argument('--batch_size', type=int, default=32, help = 'Set minibatch size')
    parser.add_argument('--level', type=int, default=1, help = "Max Genre Level hierarchy")
    parser.add_argument('--use_static', action='store_true', default=False, help = "Use static embeddings")
    parser.add_argument('--sequence_length', type=int, default=100, help = "Maximum sequence length")
    parser.add_argument('--epochs', type=int, default=60, help = "Number of epochs to run")
    parser.add_argument('--activation_th', type=float, default=0.5, help = "Activation Threshold of output")
    parser.add_argument('--lstm_units', type=int, default=700, help = "Number of units in LSTM")
    parser.add_argument('--num_filters', type=int, default=500, help = "Number of filters in CNN and Capsule")
    parser.add_argument('--adjust_hierarchy', type=str, default='None', choices=['None','semi_transitive', 'transitive', 'restrictive', 'threshold'],
     help = "Postprocessing hierarchy correction")
    parser.add_argument('--correction_th', type=float, default=0.5, help = "Threshold for Hierarchy adjust, in threshold type")
    parser.add_argument('--init_layer', action='store_true', default=False, help = "Init final layer with cooccurence")
    parser.add_argument('--iterations', type=int, default=3, help = "Number of iterations for training")
    parser.add_argument('--embed_dim', type=int, default=300, help = "Embedding dim size")
    parser.add_argument('--use_early_stop', action='store_true', default = False , help = 'Activate early stopping')
    parser.add_argument('--learning_decay', type=float, default = 1., help = 'Use decay in learning, 1 is None')
    parser.add_argument('--learning_rate', type = float, default = 0.0005, help = 'Set learning rate of network')
    parser.add_argument('--execute_all', action='store_true', default = False, help = 'Executes evaluation on every level of hierarchy')
    parser.add_argument('--whitespace_sep', action='store_true', default = False, help = 'Uses whitespace seperation instead of spacy')
    parser.add_argument('--use_low_freq', action='store_true', default = False, help = 'Filter low frequency words from dataset')
    parser.add_argument('--use_dev', action ='store_true', default = False, help = 'Uses dev set')
    parser.add_argument('--save_model', action='store_true', default = False, help = 'Saves model')
    parser.add_argument('--use_cv', action = 'store_true', default = False, help = 'Uses crossvalidation')
    parser.add_argument('--activations_benchmark', action = 'store_true', default = False, help = 'Test classifier for different activations')
    parser.add_argument('--store_output', action = 'store_true', default = False, help = 'Store the output of a classifier in readable format')

    args = parser.parse_args()
    args.num_split = -1
    import json
    params = vars(args)
    print(json.dumps(params, indent = 2))
    run()

def run():
    """
    Execution pipeline for each mode
    """
    classifier = args.classifier

    #used for training the model on train and dev, executes only once, simpliest version
    if args.mode =='train':
        if args.use_cv:
            init_data(dev = False)
            results = train_test_cv(data['vocabulary'])
            for key in results:
                print((key + '\n' + str(results[key][0]) + '\n' + str(results[key][1]) + '\n' + str(results[key][2])).replace(".", ","))
        else:
            init_data(dev = args.use_dev)
            model = create_model(preload = False)
            train(model,X_train = data['X_train'], y_train = data['y_train'], X_test = data['X_test'], y_test = data['y_test'],
             early_stopping = args.use_early_stop, validation = args.use_dev, save = args.save_model)
            results = test(model, data_l = data['X_test'], label = data['y_test'])

            #save_scores([results])

    elif args.mode =='gridsearch':
        init_data(dev = False)
        best_params = gridsearch()
        print("Best parameters: ", best_params[0], "; Precision: " , best_params[1], ". Other parameters: Sequence Length: ", args.sequence_length,
        "init_layer: ", args.init_layer, "; embed_dim: ", args.embed_dim, "; batch-size: ", args.batch_size, "; adjust_hierarchy: ", args.adjust_hierarchy)

    #create graph of model, not tested for capsule
    elif args.mode == 'plot':
        create_machine_plots(args.classifier)

    elif args.mode == 'train_n_models':
        results = []
        if args.use_cv:
            init_data(dev = False)
            for i in range(args.iterations):
                avg_result_prec, avg_result_recall, avg_result_f, avg_result_acc = train_test_cv(data['vocabulary'])
                print((str(avg_result_prec) +  '\n' + str(avg_result_recall) + '\n' +str(avg_result_f) + '\n' + str(avg_result_acc)).replace(".", ","))
                results.append([avg_result_prec, avg_result_recall, avg_result_f, avg_result_acc])
            else:
                init_data(dev = args.use_dev)
                results = []
                for i in range(args.iterations):
                    model = create_model(preload = False)
                    train(model, X_train = data['X_train'], y_train = data['y_train'], X_test = data['X_test'], y_test = data['y_test'],
                     early_stopping = args.use_early_stop, validation = args.use_dev, save = args.save_model)
                    result = test(model, data_l = data['X_test'], label = data['y_test'])
                    results.append(result)
            save_scores(results)
    else:
        print("No mode selected, aborting program")
        return

    print(args.filename)
    K.clear_session()


def create_model(preload = True):
    """
    General method to create model based on user arguments
    """
    general_name = ("__batchSize_" + str(args.batch_size) + "__epochs_" + str(args.epochs)
    + "__sequenceLen_" + str(args.sequence_length)  + "__activThresh_" + str(args.activation_th) + "__initLayer_"
    + str(args.init_layer) + "__adjustHier_" + str(args.adjust_hierarchy) +  "__correctionTH_"
    + str(args.correction_th) + "__learningRate_" + str(args.learning_rate) + "__decay_"
    + str(args.learning_decay) + "__lang_" + args.lang)
    if args.classifier == 'lstm':
        args.filename = ('lstm__lstmUnits_' + str(args.lstm_units) + general_name)
        return model_lstm(preload)
    elif args.classifier == 'cnn':
        args.filename = ('cnn__filters_' + str(args.num_filters) + general_name)
        return model_cnn(preload)
    elif args.classifier == 'character_cnn':
        args.filename = ('character_cnn__filters_' + str(args.num_filters) + general_name)
        return model_character_cnn(preload)
    elif args.classifier == 'capsule':
        args.filename = ('capsule__filters_' + str(args.num_filters) + general_name)
        return model_capsule(preload)
    print(args.filename)


def init_data(dev, outlier = False):
    """
    Initilizes the data(splits) and vocabulary
    """
    global data
    use_spacy = not args.whitespace_sep
    use_low_freq = args.use_low_freq
    if dev:
        X_train, y_train, X_dev, y_dev, X_test, y_test, vocabulary, vocabulary_inv =load_data(spacy = use_spacy, lowfreq = use_low_freq,
         max_sequence_length =  args.sequence_length, type = args.lang, level = args.level, dev = dev, cv = False, classifier = args.classifier)
        data['X_dev'] = X_dev
        data['y_dev'] = y_dev
    else:
        X_train, y_train, X_test, y_test, vocabulary, vocabulary_inv = load_data(spacy = use_spacy, lowfreq = use_low_freq,
         max_sequence_length =  args.sequence_length, type = args.lang, level = args.level, dev = dev, cv = args.use_cv, classifier = args.classifier)

    data['X_train'] = X_train
    data['y_train'] = y_train
    
    if len(X_test) == 0:
        print("Set Test data to training data since no other data is available...")
        data['X_test'] = X_train
        data['y_test'] = y_train
    else:
        data['X_test'] = X_test
        data['y_test'] = y_test
    data['vocabulary'] = vocabulary
    data['vocabulary_inv'] = vocabulary_inv

if __name__ == '__main__':
    main()
