# app.py
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

predict_pipeline = PredictPipeline()

@app.route('/', methods=['GET'])
def home():
    # Fetch all movie titles to display in the dropdown
    # This assumes predict_pipeline has a method to return all movie titles
    all_movie_titles = predict_pipeline.get_all_movie_titles()
    return render_template('home.html', movie_titles=all_movie_titles)

@app.route('/predict', methods=['POST'])
def predict():
    movie_title = request.form['movie_title']
    movie_id = predict_pipeline.get_movie_id_by_title(movie_title)
    
    if movie_id is None:
        return "Movie not found", 404
    
    similar_movies_ids = predict_pipeline.find_similar_movies(movie_id)
    similar_movies_titles = [predict_pipeline.get_title_by_movie_id(id) for id in similar_movies_ids]
    return render_template('results.html', movie_titles=similar_movies_titles)


@app.route('/user-recommend', methods=['GET', 'POST'])#
def user_recommend():
    if request.method == 'POST':
        user_id = request.form.get('user_id', type=int)
        recommendations = predict_pipeline.get_user_recommendations(user_id)
        return render_template('user_recommendations.html', recommendations=recommendations)
    return render_template('user_recommend_form.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
