from flask import Flask, request,jsonify, render_template
import requests
from pymongo import MongoClient
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import pandas as pd
import numpy as np
from flask_bcrypt import Bcrypt
import os
from dotenv import load_dotenv



# Load model and dataset
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('cleaned.csv')

# Convert vector column from string to NumPy arrays
df['vector'] = df['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

client = MongoClient(os.getenv('MONGO_URI'))
db = client['MovieDB']
movies_collection = db['Movies']
#connecting the cloud server
client2 = MongoClient(os.getenv('MONGO_URI'))
db2 = client2['UserDatabase']
collection = db2['Users']

bcrypt = Bcrypt(app)

def get_vector(text, model):
    """Convert text into vector using word embedding model"""
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def fetch_poster(movie_id):
    
    api_key =  os.getenv('TMDB_API_KEY')
    base_url = 'https://api.themoviedb.org/3/movie/'
    
    response = requests.get(f"{base_url}{movie_id}?api_key={api_key}&language=en-US")
    
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path', None)
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    return "https://via.placeholder.com/500x750?text=No+Image"

def recommend_movies_optimized(description, df, model, top_n=12):
    """Recommend top movies based on input description"""
    query = simple_preprocess(description)
    query_vector = get_vector(query, model)
    
    # Compute similarity for all movies in a single operation
    movie_vectors = np.vstack(df['vector'].values)  # Convert all movie vectors into a NumPy matrix
    similarities = cosine_similarity([query_vector], movie_vectors)[0]  # Compute similarities

    # Get top N movies
    top_indices = similarities.argsort()[-top_n:][::-1]  # Get indices of top matches
    top_movies = df.iloc[top_indices][['title','id']].copy()
    top_movies['similarity'] = similarities[top_indices]
    top_movies['poster'] = top_movies['id'].apply(fetch_poster)

    return top_movies


@app.route('/')
def sign():
    return render_template('sign-in.html')


@app.route('/signup', methods =['GET','POST'])
def index():
    
    if request.method =='POST':
        name = request.form.get("username")
        mobnumber = request.form.get("mobnumber")
        email = request.form.get("usermail")
        password = request.form.get("userpassword")

        userexist = collection.find_one({"email":email})
        if userexist:
            message = "Email already registered..!"
            return render_template('sign-in.html', message=message)
        else:
            hashed_pass = bcrypt.generate_password_hash(password).decode('utf-8')
            collection.insert_one({"name":name, "mobile number":mobnumber, "email":email, "password": hashed_pass})
            message1="Signed in successfully"
    return render_template('index.html',message=message1)

@app.route('/login.html', methods=['GET','POST'])
def login():
    if request.method =="POST":
        logmail = request.form.get("usermail")
        
        logpass = request.form.get("userpassword")
        
        checkmail = collection.find_one({"email":logmail})
        
        if checkmail and bcrypt.check_password_hash(checkmail.get("password"),logpass):
            message ="Logged in successfully"
            return render_template('index.html',message=message)
    return render_template('login.html')

@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/suggest')
def suggest():
    search_query = request.args.get('query', '')
    results = movies_collection.find({'title': {'$regex': search_query, '$options': 'i'}}).limit(10)
    movies = [movie['title'] for movie in results]
    return jsonify(movies)

@app.route('/recommend', methods=['GET'])
def recommend():
    desc = request.args.get('query', '')
    top_movies = recommend_movies_optimized(desc, df, model)
    return jsonify(top_movies.to_dict(orient='records'))  

if __name__ == "__main__":
    app.run(debug=True)
