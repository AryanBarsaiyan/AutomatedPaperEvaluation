from flask import Flask, request
from flask_cors import CORS
from nltk.corpus import stopwords
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import getpass
import os
from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

# llm = OpenAI(model="gpt-3.5-turbo-instruct")

app = Flask(__name__)
CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

# Use NLTK's English stop words list
stop_words = set(stopwords.words('english'))

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    tokens = nltk.word_tokenize(text.lower().translate(remove_punctuation_map))
    # Filter out stop words
    tokens = [token for token in tokens if token not in stop_words]
    return stem_tokens(tokens)

vectorizer = TfidfVectorizer(tokenizer=normalize, token_pattern=None)

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]

import re

def extract_answers(response_string):
    # Use regular expression to split the response string based on numbering format
    answer_list = re.split(r'\d+\.\s+', response_string)
    
    # Filter out empty strings and strip any leading/trailing whitespace
    clean_answers = [answer.strip() for answer in answer_list if answer.strip()]
    
    return clean_answers

def extract_score(response_string):
    # Use regular expression to extract the score from the response string
    score_match = re.search(r'Score : (\d+)', response_string)
    
    if score_match:
        return int(score_match.group(1))
    
    return -1


# Initialize the OpenAI model
llm = OpenAI(model="gpt-3.5-turbo-instruct")

def getEditorialAnswers(question, editorialAnswer):
    # Generate 10 editorial answers with the provided question and editorial answer
    prompt_value = f"Generate 10 editorial answers for the question: '{question}' and provide responses that are similar in length and content to the provided editorial answer:\n\n'{editorialAnswer}'. Ensure that the generated answers maintain the same level of detail and complexity as the reference editorial answer."
    # print("Prompt:", prompt_value)
    message = llm.invoke(prompt_value,max_tokens=2048)
    print("Response:", message)

    # Extract answers from the message content
    answers = extract_answers(message)
    # print("Extracted answers:", answers)
    return answers

def gptAnswerAnalysis(question, userAnswer, editorialAnswer):
    # Construct the editorial answer with pt and its score
    prompt_value = f"Break down the editorial answer into key points with their importance scores for the question: '{question}' such that collective importance score must be 100. The editorial answer is editorialAnswer :\n\n'{editorialAnswer}'."
    
    # Invoke the model to generate response
    result = llm.invoke(prompt_value, max_tokens=2048)
    editorialAnswerPoints = result
    
    print ("Editorial Answer Points:", editorialAnswerPoints)
    
    
    
    # Construct the prompt
    prompt_value = f"Given the question: '{question}', the user provided the following answer:\n\n userAnswer: '{userAnswer}'. The editorial answer with assigned importance score for this question is:\n\n editorialAnswerPoints: '{editorialAnswerPoints}'. Evaluate the userAnswer strictly against the editorialAnswerPoints. Precisely check the userAnswer against the editorialAnswerPoints and consider its importance score, and provide a score between 0 to 100 based on the importance points covered by userAnswer. If a point is directly covered by userAnswer, assign the corresponding importance score; otherwise, assign 0. And also provide the reasoning explanation for the assigned score and the response format should be like this: 'Score : ___' \n\n.Explaination: '___'"

    # print("Prompt:", prompt_value)   
    
    # Invoke the model to generate response
    result = llm.invoke(prompt_value, max_tokens=2048)
    print("Response:", result)
    
    # output:
    # Response: 

    # Score : 0
    # Explanation: The user answer does not directly mention any of the points provided in the editorial answer. It mostly talks about thrashing, which is not directly related to the concept of processes and process table.
    
    # extract the score from the message
    score = extract_score(result)
    
    print("MyScore:", score)


    # Extract the score from the message
    # score = extract_score(message)
    
    # # Format the response message
    # response_message = f"Your score is {score}"
    
    # print("Response:", response_message)
    
    # return response_message 

@app.route('/', methods=['GET', 'POST']) 
def matchCaller():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get("Question")
        user_answer = data.get("userAnswer")
        editorial_answer = data.get("editorialAnswer")

        # editorial_answers = getEditorialAnswers(question, editorial_answer)
        # print the size of the editorial answers
        # print("Editorial Answers:", len(editorial_answers))

        ans = gptAnswerAnalysis(question, user_answer, editorial_answer)
        # print("Similarity:", ans)

        max_sim = cosine_sim(user_answer, editorial_answer)
        # for answer in editorial_answers:
        #     print(answer)
        #     sim = cosine_sim(user_answer, answer)
        #     sim = min(1.0, max(0.0, sim))
        #     print("Similarity:", sim)
            
        #     if sim > max_sim:
        #         max_sim = sim

        return str(max_sim)
    else:
        return "Welcome to the backend!"

if __name__ == "__main__":
    app.run(debug=True)
