import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)




with open("intents.json") as file:
    data = json.load(file)

try:
    with open("other/data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)  #to bring it to the root word
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    #print(docs_x)
    #print(docs_y)

    #Data Preprocessing

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]    # Main word list
    words = sorted(list(set(words)))    # removing duplicates

    labels = sorted(labels)
    #print(words)


    ''' Using One Hot Encoding to convert string into numerals,
        In our case we will be stating whether that word exist in our main word list(words)
        if exists it will append 1 in bag else it will append 0
        This will be used as an input to the Neural Network'''

    training = []   #Set up training output
    output = []
    out_empty = [0 for _ in range(len(labels))] #initializes every tag by 0

    for x, doc in enumerate(docs_x):
        bag = []    #bag of one hot encoded words
        
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
            
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 #it will add 1 to that corresponding tag

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("other/data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)


# Creating the model

''' Our model basically predicts which tag that we should take a response from to give to the user
    softmax function gives a probability rating to each tag and the tag having highest probability is
    then given as an output to the user'''

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])  #input layer
net = tflearn.fully_connected(net, 8)   # hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")    #output layer   softmax gives the probability for each output(neuron)
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("other/model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("other/model.tflearn")


def bag_of_words(sentence, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

@app.route("/", methods=['GET', 'POST'])
def chat():
   
    inp = request.form.get('text_input')
    output_text=""
    if request.method == "POST":

       
        result = model.predict([bag_of_words(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            
            #print(random.choice(responses))
            output_text=random.choice(responses)
            #print(output_text)
        else:
            output_text="I didn't understand that, Please ask another question..."

    return render_template("index.html",content=output_text)

#chat()

if __name__ == "__main__":
    app.debug = True
    app.run()