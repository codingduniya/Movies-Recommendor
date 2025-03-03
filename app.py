from flask import Flask, request,jsonify, render_template, make_response, redirect, url_for
from dotenv import load_dotenv
import os
import requests
from pymongo import MongoClient
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import pandas as pd
import numpy as np
from flask_bcrypt import Bcrypt
import certifi
from flask_mail import *





# Load model and dataset
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('folder_for_csv\cleaned.csv')

# Convert vector column from string to NumPy arrays
df['vector'] = df['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.getenv('FLASK_MAIL')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASS')

mail = Mail(app)
client = MongoClient(os.getenv('MONGO_URI'),tlsCAFile=certifi.where())
db = client['MovieDB']
movies_collection = db['Movies']
#connecting the cloud server
client2 = MongoClient(os.getenv('MONGO_URI'),tlsCAFile=certifi.where())
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


@app.route('/', methods=['GET','POST'])
def sign():
    user_data = request.cookies.get("mail")
    if request.method =='POST' :
        return render_template('sign-in.html')
    elif user_data:
        
        return render_template('index.html',message1="Welcome to you again "+(request.cookies.get("name")))
    else:
        return render_template('home.html')


@app.route('/signup', methods =['GET','POST'])
def index():
    
    if request.method =='POST':
        name = request.form.get("username")
        mobnumber = request.form.get("mobnumber")
        email = request.form.get("usermail")
        password = request.form.get("userpassword")

        userexist = collection.find_one({"email":email})
        if userexist:
            message1 = "Email already registered..!"
            return render_template('sign-in.html', message1=message1)
        else:
            hashed_pass = bcrypt.generate_password_hash(password).decode('utf-8')
            collection.insert_one({"name":name, "mobile number":mobnumber, "email":email, "password": hashed_pass})
            message1="Signed in successfully"
            response = make_response(render_template('index.html',message1=message1))
            response.set_cookie("name",name,max_age=60*60*24*30, path="/")
            response.set_cookie("mail", email, max_age=60*60*24*30, path = "/")
            msg = Message("Welcome To MooovieRec" , sender="mooovierecnoreply@gmail.com", recipients=[email])
            msg.body = "Hello "+ name + " thank you for registering in the MooovieRec\nThank you for using this website. \nYour password is "+password+" \nplease do not share the password with any one"
            mail.send(msg)
            return response

@app.route('/Logout', methods=['GET'])
def logout():
    response = make_response(redirect(url_for('sign')))
    response.delete_cookie("name", path='/')
    response.delete_cookie("mail", path='/')
    return response


@app.route('/login.html', methods=['GET','POST'])
def login():
    if request.method =="POST":
        logmail = request.form.get("usermail")
        
        logpass = request.form.get("userpassword")
        
        checkmail = collection.find_one({"email":logmail})
        
        if checkmail and bcrypt.check_password_hash(checkmail.get("password"),logpass):
            message1 ="Logged in successfully :)"
            return render_template('index.html',message1=message1)
        else:
            return render_template('login.html', message1 = "Incorrect Username Or Password !!")
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

@app.after_request
def add_no_cache_headers(response):
    if request.endpoint in ['index', 'dashboard', 'profile']:  # Protect specific pages
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.route('/terms.html', methods=['GET'])
def terms():
    return render_template('terms.html')


if __name__ == "__main__":
    app.run(debug=True, port=3000)
