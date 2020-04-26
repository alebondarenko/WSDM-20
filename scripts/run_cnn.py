import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import pickle
import argparse
import re
import string
import subprocess
import json
import os

np.set_printoptions(suppress=True)

def test(model, data, threshold=0.5):
    """
    Tests a neural network on the given data
    """
    print("Testing Model...")
    output = model.predict(data, batch_size = 32) # returns probabilities that the question is comparative
    #binary_output = np.array(output, copy = True) # uncomment to return a binary 0 and 1 output
    #for pred_i in output:
    #    pred_i[pred_i >=threshold] = 1
    #    pred_i[pred_i < threshold] = 0
    return output

def model_cnn(filename):
    """
    Creates CNN or loads it if available
    """
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.isfile(filepath):
        print ("Loading model...")
        model = load_model(filepath)
        model.summary()
        return model
    else:
        print("Model not found, Aborting...")
        exit(1)

def spacy_tokenizer_basic_pos(sentence):
    sentence_n = []
    sentence_pos = pos(mst.process(sentence))
    return sentence_pos

def pos(l):
    dd=[]
    for d1 in l:
        l2 = d1.get('analysis', [])
        l3 = d1.get('text', [])    #using original words
        if l2 != []:
            dd.append(l3)
            grammems = []
            for d2 in l2:
                if 'gr' in d2:
                    grammems.append(d2['gr'])
                    grammems_str = ' '.join(grammems).lower()
            pos = ''
            if re.search('comp', grammems_str): pos = 'comp'
            elif re.search('supr', grammems_str): pos = 'supr'
            else: pos = grammems[0].split('=')[0].split(',')[0]
            dd.append(pos)
        elif 'analysis' in d1:
            dd.append(l3)
            dd.append('latin')
    return ' '.join(dd).lower()

class Mystem(object):

    def __init__(self):
        self._proc = None

    def _start(self):
        self._proc = subprocess.Popen(
           "./mystem --format json -cgi --eng-gr".split(),
           stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _getProc(self):
        if self._proc is None:
            self._start()
        return self._proc

    def process(self, text):
        p = self._getProc()
        p.stdin.write(text.strip().encode('utf8'))
        p.stdin.write('\n'.encode('utf8'))
        p.stdin.flush()
        return json.loads(p.stdout.readline().decode('utf8'))
mst = Mystem()

def load_data(filename, vocabulary_path):
    """
    preprocessing of test data with no labels
    """
    file = open(filename, 'r')
    test_data = file.readlines()[1:] # Careful, tested file has a header
    file.close()
    questions = [x.split('\t')[-1].strip() for x in test_data] #Careful here, this is just because the file I tested also has the labels
    y_test = [float(y.split('\t')[0].strip()) for y in test_data] #Careful here, this is just because the file I tested also has the labels
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    x_text = [regex.sub('', s) for s in questions]
    x_text = [spacy_tokenizer_basic_pos(sentence) for sentence in x_text]
    x_text = [s.split(" ") for s in x_text]
    vocabulary = pickle.load(open(vocabulary_path, 'rb'))
    UNSEEN_STRING = "-EMPTY-"
    x_text = [[word if word in vocabulary else UNSEEN_STRING for word in sentence] for sentence in x_text]
    x_text = np.asarray([[vocabulary[word] for word in sentence] for sentence in x_text])
    x_text = pad_sequences(x_text, maxlen=15, dtype='str', padding = 'post', truncating ='post')
    #x_text = np.asarray([[vocabulary[word] for word in sentence] for sentence in x_text])
    return x_text, y_test, questions

def save_results(predictions, data, filename):
	"""
	save predictions to a given file
	"""
	float_formatter = lambda x: "%.10f" % x
	with open(filename, 'w') as file:
		file.write('Comparative_Confidence\tquestion\n')
		for i in range(len(predictions)):
			file.write(str(float_formatter(predictions[i])) + '\t' + str(data[i]) + '\n')
	file.close()

def main():
    parser = argparse.ArgumentParser(description="CNN")
    parser.add_argument('--input_path', type=str, help="Input file with sentences the cnn should classify.")
    parser.add_argument('--cnn_path', type=str, help="Path of the CNN.")
    parser.add_argument('--vocabulary_path', type=str, help="Path of the vocabulary for CNN")
    parser.add_argument('--threshold', type=float, help= "Threshold for activation of CNN")
    parser.add_argument('--output_path', type=str, help='Output path with classification results')
    args = parser.parse_args()
    data, y_test, questions = load_data(args.input_path, args.vocabulary_path)
    model = model_cnn(args.cnn_path)
    predictions = test(model, data, args.threshold)
    #print(classification_report(y_true=y_test, y_pred=predictions))
    save_results(predictions, questions, args.output_path)


if __name__ == '__main__':
    main()

