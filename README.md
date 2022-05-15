# Build Chatbot With Python and Integrate It on Telegram

Authors: Talipzhanuly Bilalidin and Telman Alisher

Domain customer support 

# Introduction
Were you ever curious as to how to build a talking ChatBot with Python and also have a conversation with your own personal AI?

As the topic suggests we are here to help you have a conversation with your AI today. To have a conversation with your AI, you need a few pre-trained tools which can help you build an AI chatbot system. In this article, we will guide you to combine speech recognition processes with an artificial intelligence algorithm.

Natural Language Processing or NLP is a prerequisite for our project. NLP allows computers and algorithms to understand human interactions via various languages. In order to process a large amount of natural language data, an AI will definitely need NLP or Natural Language Processing. Currently, we have a number of NLP research ongoing in order to improve the AI chatbots and help them understand the complicated nuances and undertones of human conversations.

![image](https://user-images.githubusercontent.com/46298232/168478815-44af019e-2c3d-4a26-8f1f-eda3e556bdcc.png)


Chatbots are nothing but applications that are used by businesses or other entities to conduct an automatic conversation between a human and an AI. These conversations may be via text or speech. Chatbots are required to understand and mimic human conversation while interacting with humans from all over the world. From the first chatbot to be created ELIZA to Amazon’s ALEXA today, chatbots have come a long way. In this tutorial, we are going to cover all the basics you need to follow along and create a basic chatbot that can understand human interaction and also respond accordingly. We will be using speech recognition APIs and also pre-trained Transformer models.

# Description


# System models
The first and foremost thing before starting to build a chatbot is to understand the architecture. For example, how chatbots communicate with the users and model to provide an optimized output.
![image](https://user-images.githubusercontent.com/46298232/168479340-ede93172-bbf8-4122-aad1-9bead10033c2.png)
In the above image, we have Training Data/ Corpus, Application DB, NLP Model, and a Chat Session.

## Training Data/ Corpus

Before we can begin to think of any coding, we need to set up an intents JSON file that defines certain intentions that could occur during the interactions with our chatbot. To perform this we would have to first create a set of tags that users queries may fall into. For example:
* A user may wish to know the price of our education center, therefore we create an intention labeled with a tag called `price`
* A user may wish to know the courses of our education center, therefore we create an intention labeled with the tag `courses`
* etc etc
For each of the tags that we create, we would have to specify patterns. Essentially, this defines the different ways of how a user may pose a query to our chatbot. For instance, under the `price` tag, a user may ask price of courses  in a variety of ways — _“Сколько стоят курсы?”, “Можно узнать ваш прайс лист?”, “Стоимость?”._ 
**(russian)**,  _"How much cost courses?","Can I get your price list?",_ **(english)**

The chatbot would then take these patterns and use them as training data to determine what someone asking for our chatbot's price would look like so that it could adapt to the different ways someone may ask to know costs our education center . Therefore, users wouldn’t have to use the exact queries that our chatbot has learned. It could pose the question as _“Сколько стоят курсы?”("How much cost courses?)_ and our chatbot would be able to infer that the user wants to know the price of our education centre and then it would provide costs.
> Note: Our bot is not going to be super-duper intelligent so it would not always recognize what we are saying, but with enough examples, it would do a pretty decent job at deciphering.

Within this intents JSON file, alongside each intents tag and pattern, there will be responses. However, for our  chatbot, these responses are not going to be generated. What this means is that our patterns aren’t going to be as free-flowing as the patterns users can ask (it will not adapt), instead, the responses will be using static responses that the chatbot will return when posed with a query.

![image](https://user-images.githubusercontent.com/46298232/168480546-d792a1a1-4f77-4e8b-9255-7344e0a73a46.png)

# Neural Network
![image](https://user-images.githubusercontent.com/46298232/168482472-c53d25b7-9b95-407a-8e60-9a74c7329eeb.png)
Image Credit : alexlenail.me/NN-SVG/ 

It is a deep learning algorithm that resembles the way neurons in our brain process information (hence the name). It is widely used to realize the pattern between the input features and the corresponding output in a dataset. Here is the basic neural network architecture 

The purple-colored circles represent the input vector, xi where i =1, 2, ….., D which is nothing else but one feature of our dataset. The blue-colored circles are the neurons of hidden layers. These are the layers that will learn the math required to relate our input with the output. Finally, we have pink-colored circles which form the output layer. The dimension of the output layer depends on the number of different classes that we have. For example, let us say we have 5x4 sized dataset where we have 5 input vectors, each having some value for 4 features: A, B, C, and D. Assume that we want to classify each row as good or bad and we use the number 0 to represent good and 1 to represent bad. Then, the neural network is supposed to have 4 neurons at the input layer and 2 neurons at the output.
Okay, so now that you have a rough idea of the deep learning algorithm, it is time that you plunge into the pool of mathematics related to this algorithm.
![image](https://user-images.githubusercontent.com/46298232/168482623-e3484094-e7e3-478c-baa0-f8513be0f2b1.png)

Neural Network algorithm involves two steps:
1. Forward Pass through a Feed-Forward Neural Network
2. Backpropagation of Error to train Neural Network

### 1. Forward Pass through a Feed-Forward Neural Network

This step involves connecting the input layer to the output layer through a number of hidden layers. The neurons of the first layer (l=1) receive a weighted sum of elements of the input vector (xáµ¢) along with a bias term b, as shown in Fig. 2. After that, each neuron transforms this weighted sum received at the input, a, using a differentiable, nonlinear activation function h(·) to give output z.

![image](https://user-images.githubusercontent.com/46298232/168482765-63103ac7-d14c-4ef5-99e9-fdbf6334a2d8.png)
Hidden layers of neural network architecture.
Image Credit : alexlenail.me/NN-SVG/ 

For a neuron of subsequent layers, a weighted sum of outputs of all the neurons of the previous layer along with a bias term is passed as input. This has been represented in Fig. 3. The layers of the subsequent layers to transform the input received using activation functions.

This process continues till the output of the last layer’s (l=L) neurons has been evaluated. These neurons at the output layer are responsible for identifying the class the input vector belongs to. The input vector is labeled with the class whose corresponding neuron has the highest output value.

Please note that the activation functions can be different for each layer. The two activation functions that we will use for our ChatBot, which are also most commonly used are Rectified Linear Unit (ReLu) function and Softmax function. The former will be used for hidden layers while the latter is used for the output layer. The softmax function is usually used at the output for it gives probabilistic output. The ReLU function is defined as:

![image](https://user-images.githubusercontent.com/46298232/168482864-5d201e6e-a4df-41fc-b5b9-7fc951366519.png)

and the Softmax function is defined as:

![image](https://user-images.githubusercontent.com/46298232/168482881-d633b726-e097-4711-bff0-b1c768cbd501.png)

### 2. Backpropagation of Error to train Neural Network

This step is the most important one because the original task of the Neural Network algorithm is to find the correct set of weights for all the layers that lead to the correct output and this step is all about finding those correct weights and biases. Consider an input vector that has been passed to the network and say, we know that it belongs to class A. Assume the output layer gives the highest value for class B.  There is therefore an error in our prediction. Now, since we can only compute errors at the output, we have to propagate this error backward to learn the correct set of weights and biases.

Let us define the error function for the network:

![image](https://user-images.githubusercontent.com/46298232/168482924-9f105686-2695-4d45-b340-fe9acf010c19.png)
![image](https://user-images.githubusercontent.com/46298232/168482928-3d7c63c6-913e-4156-8bbc-112735bb2683.png)

### NLP Model
The NLP model is a Deep-Learning model. As per [SAS](https://www.sas.com/en_in/insights/analytics/what-is-natural-language-processing-nlp.html), Natural language processing helps computers communicate with humans in their own language and scales other language-related tasks. For example, NLP makes it possible for computers to read text, hear speech, interpret it, measure sentiment, and determine which parts are important.

### NLTK

NLTK (Natural Language Toolkit) is the primary platform for building Python projects to work with human language information. It gives simple-to-utilize interfaces to more than 50 corpora and lexical resources, for example, WordNet, alongside the setup of text-handling libraries for classification, tokenization, stemming, tagging, parsing, and semantic thinking, and wrappers for industrial-strength NLP libraries.

### Text pre-processing with NLTK
The principle issue with text information is that it is all in strings (a group of text). However, machine learning calculations need a type of numerical element vector to perform the task. So before we start with any NLP project, we have to pre-process it to make it perfect for work. Fundamental text pre-handling incorporates:

* Changing over the whole content into uppercase or lowercase, with the goal that the calculation doesn’t treat the same words in different contexts.
* Tokenization is a process in which the text of strings is converted into a list of tokens. There is a sentence tokenizer that can be used to find the list of sentences and word tokenizer that can be used to find the list of words in strings.
* Removing noise will remove everything that isn’t a standard letter or number, like a punctuation mark, extra spaces, etc.
* Removing stop words: Stop words are words that are commonly used (such as the, a, an, in, etc.) that have little value in selecting the matching phrase according to a user query.
* Stemming: It is a process of reducing a derived form of a word to its stem, base, or root form. For example, if we stem the following words: walks, walking, walked, then the stem word would be a single word, walk.
* Lemmatization: A transformed version of stemming is lemmatization. The significant difference between these is that stemming operates on a single word without the knowledge of the context and can often create a non-existing word. In contrast, after lemmatization, we will get a valid word that has meaning in the dictionary. Lemmatization is based on the part of speech of a word that should be determined to get the correct lemma of the word. An example of lemmatization is that is, am, and are are forms of the verb to be; therefore, their lemma is be.

### Bag of words

After the initial cleaning and processing phase, we need to transform the text into a meaningful vector (or cluster) of numbers. According to [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/#:~:text=A%20bag-of-words%20is,the%20presence%20of%20known%20words) by Jason Brownlee,

> “The bag of words is a representation of text that describes the occurrence of words inside a document. It includes two things:
> 1. A vocabulary of known words
> 2. A measure of the presence of all the known words”

In the bag-of-words model, a text (such as a sentence or a document) is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity. The main reason behind using it is to check whether the sentence is similar in content or not with the document.

Suppose the vocabulary contains the words: {knowing, must, do, enough}. And we have the following sentence: “Knowing is not enough; we must apply.” Then bag-of-words representation for this would have the resulting vector: [1, 1, 0, 1].

### Application DB
Application DB is used to process the actions performed by the chatbot.

### Chat Session/ User Interface
A chat session or User Interface is a frontend application used to interact between the chatbot and end-user.

# Chatbot creation
Now we know some terminologies used in chatbot creation and have a fair idea of the NLP process. We are going to create and deploy a chatbot in Telegram that will give you information about the company and reduce the burden on managers.

###  Importing the necessary libraries
# import necessary libraries
```
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from keras.models import load_model

# create an object of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# importing the GL Bot corpus file for pre-processing

words=[]
classes = []
documents = []
ignore_words = ['?', '!',',','.']
data_file = open("intents.json").read()
intents = json.loads(data_file)
print(intents)
```
### Data pre-processing
Data preprocessing can refer to the manipulation or dropping of data before it is used in order to ensure or enhance performance, and is an important step in the data mining process. It takes the maximum time of any model building exercise which is almost 70%

```
# preprocessing the json data
# tokenization
nltk.download('punkt')
nltk.download('wordnet')
for intent in intents['intents']:
    for pattern in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
```
In the above code, we are using the Corpus Data which contains nested JSON values, and updating the existing empty lists words, documents, and classes.

Tokenize or Tokenization is used to split a large sample of text or sentences into words. In the below image, I have shown the sample from each list we have created.
```
# lemmatize, lower each word and remove duplicates

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print (len(documents), "documents")

# classes = intents
print (len(classes), "classes", classes)

# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

# creating a pickle file to store the Python objects which we will use while predicting
pickle.dump(words,open('words.pkl','wb')) 
pickle.dump(classes,open('classes.pkl','wb'))
```
> Chatbot- Lemmatizer

![image](https://user-images.githubusercontent.com/46298232/168486072-77d04205-e33c-451a-b46a-9c6d6b9e1844.png)
> Output
In the above output, we have observed a total of 67 documents, 8 classes, and 102 unique lemmatized words. We have also saved the words and classes for our future use.

### Creating training data
```
# create our training data
training = []

# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
   
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle features and converting it into numpy arrays
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")
```
> Chatbot - Training Data
In the above code, we have created a bow (bag of words) for each sentence. Basically, a bag of words is a simple representation of each text in a sentence as the bag of its words.

For example, consider the following sentence “John likes to watch movies. Mary likes movies too”.
![image](https://user-images.githubusercontent.com/46298232/168486212-0b72661f-21a3-4bcc-a01b-3d26f9e01988.png)
> Bag of Words- Example
After creating the training dataset, we have to shuffle the data and convert the lists into NumPy array so that we can use it in our model.

Now, separate the features and target column from the training data as specified in the above code
### Creating a neural network model
In this step, we will create a simple sequential NN model using one input layer (input shape will be the length of the document), one hidden layer, an output layer, and two dropout layers.
```
# Create NN model to predict the responses
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot.h5', hist) # we will pickle this model to use in the future
print("\n")
print("*"*50)
print("\nModel Created Successfully!")
```
![image](https://user-images.githubusercontent.com/46298232/168486298-de15ce57-6b86-4725-a594-c2c8abd48c19.png)
> Output
The summary of the model is shown in the below image.
![image](https://user-images.githubusercontent.com/46298232/168486406-08367b37-ec18-4770-bbf7-6352059b00eb.png)
The accuracy of the above Neural Network model is almost 100% which is quite impressive. Also, we have saved the model for future use.

### Create functions to take user input, pre-process the input, predict the class, and get the response
```
def clean_up_sentence(sentence):

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
```

**Function to create bow (bag of words) using the clean sentence from the above step**
```
def bow(sentence, words, show_details=True):

    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
               
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
```
**Function to predict the target class**
```
def predict_class(sentence, model):
   
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
```
** Function to get the response from the model**
```
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
```
The following bot_initialize function will get the input of the user query from Telegram and will start processing it to generate a credible response. We should also make our chatbot sophisticated so that it can greet the user at the commencement and termination of the conversation.
```
# function to start the chat bot which will continue till the user type 'закончить'

def bot_initialize():
    print("ЧатБот: Это Ерикбурден! Ваш персональный помощник.\n\n")
    while True:
        inp = str(input()).lower()
        if inp.lower()=="Закончить":
            break
        if inp.lower()== '' or inp.lower()== '*':
            print('Пожалуйста измените свой вопрос, я вас не понимаю(')
            print("-"*50)
        else:
            print(f"Бот: {chatbot_response(inp)}"+'\n')
            print("-"*50)
```
## Activating the data-driven Telegram bot
For activating and deploying your chatbot on Telegram, you have to meet some initial prerequisites.

1. First of all, you should have an account on Telegram.
2. After creating your account on Telegram, search for BotFather, and create your chatbot as instructed by BotFather. After successfully creating your chatbot, BotFather will give you a token to authorize the bot and send requests to the Bot API. You should get a message like this: Use this token to access the HTTP API:
5302765218:AAE_WU0FEEHCXIzF6aPnaYXXXXXXXXXX .
3. After successfully getting your Telegram Bot API, write the following snippet succeeding the bot_initialize function.

```
import requests
import json

class telegram_bot():
    def __init__(self):
        self.token = "5302765218:AAE_WU0FEEHCXIzF6aPnaYKkwgI2sG1OgJg"    #write your token here!
        self.url = f"https://api.telegram.org/bot{self.token}"

    def get_updates(self,offset=None):
        url = self.url+"/getUpdates?timeout=100"   # In 100 seconds if user input query then process that, use it as the read timeout from the server
        if offset:
            url = url+f"&offset={offset+1}"
        url_info = requests.get(url)
        return json.loads(url_info.content)

    def send_message(self,text,chat_id):
        url = self.url + f"/sendMessage?chat_id={chat_id}&text={text}"
        if text is not None:
            requests.get(url)

    def grab_token(self):
        return tokens
```
```
tbot = telegram_bot()

update_id = None

# function to predict the class and get the response

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res    
    
       
while True:
    print("...")
    updates = tbot.get_updates(offset=update_id)
    updates = updates['result']
    print(updates)
    if updates:
        for item in updates:
            update_id = item["update_id"]
            print(update_id)
            try:
                message = item["message"]["text"]
                print(message)
            except:
                message = None
            from_ = item["message"]["from"]["id"]
            print(from_)

            res = chatbot_response(message)
            tbot.send_message(res,from_)
```
Finally, we made a chatbot through NLTK that can able to converse with us on Telegram when the Jupyter Notebook is running on our system

