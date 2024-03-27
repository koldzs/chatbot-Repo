import streamlit as st
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
# Comment the downloads out after you download it once so as to not affect your codes
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)

# Preprocess the data the individual customer and sales columns
cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

df = pd.DataFrame()
df['Question'] = cust[1].reset_index(drop = True)
df['Answer'] = sales[1].reset_index(drop = True)

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


df['tokenized Questions'] = df['Question'].apply(preprocess_text)

x = df['tokenized Questions'].to_list()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(x)

hist_list =[]

st.markdown("<h1 style = 'color: #31363F; text-align: center; font-family: Copperplate Gothic Semibold'>ORGANISATIONAL CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #222831; text-align: center; font-family: Copperplate Gothic Semibold '>Built By Ibrahim: </h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

chat_response, robot_image = st.columns(2)
with robot_image:
    robot_image.image('pngwing.com(90).png', caption = 'I reply all your questions', width = 500)

with chat_response:
    user_word = chat_response.text_input('Hello there you can ask your questions: ')
    def get_response(user_input):
        user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

        user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

        similarity_scores = cosine_similarity(user_input_vector, corpus) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

        most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

        return df['Answer'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response

    # create greeting list 
    greetings = ["Hey There.... I am a creation of Ehiz Danny Agba Coder.... How can I help",
                "Hi Human.... How can I help",
                'Twale baba nla, wetin dey happen nah',
                'How far Alaye, wetin happen',
                "Good Day .... How can I help", 
                "Hello There... How can I be useful to you today",
                "Hi GomyCode Student.... How can I be of use"]

    exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
    farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

    random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
    random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list

    # Test your chatbot
    # while True:
    # user_input = input("You: ")
    if user_word.lower() in exits:
        chat_response.write(f"\nChatbot: {random_farewell}!")

    elif user_word.lower() in ['hi', 'hello', 'hey', 'hi there']:
        chat_response.write(f"\nChatbot: {random_greetings}!")

    elif user_word == '':
        chat_response.write('')
        
    else:   
        response = get_response(user_word)
        chat_response.write(f"\nChatbot: {response}")

 # history starting 
        hist_list.append(user_word)



# Save the history of the texts 
with open('history.txt', 'a') as file:
        for item in hist_list:
            file.write(str(item) + '\n')
            file.write(response)    

import csv
files = 'history.txt'
with open(files) as f:
  reader = csv.reader(f)
  data = list(reader)

history = pd.Series(data)
st.sidebar.subheader('Chat History', divider = True)
st.sidebar.write(history)



# st.image('imgbin_mortgage-loan-personal-finance-financial-services-png.png', use_column_width= True)

st.header('Project Background Information', divider = True)
st.write("An organisation chatbot that uses Natural Language Processing (NLP) to preprocess company's Frequently Asked Questions(FAQ), and provide given answers to subsequently asked questions that pertains to an existing question in the FAQ.")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)



