# app.py
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
predict_pipeline = PredictPipeline()

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    movie_title = request.form['movie_title']
    movie_id = predict_pipeline.get_movie_id_by_title(movie_title)
    
    if movie_id is None:
        return "Movie not found", 404
    
    similar_movies_ids = predict_pipeline.find_similar_movies(movie_id)
    similar_movies_titles = [predict_pipeline.get_title_by_movie_id(id) for id in similar_movies_ids]
    return render_template('results.html', movie_titles=similar_movies_titles)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
