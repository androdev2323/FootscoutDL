from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from duckduckgo_search import DDGS
import logging
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data and model
logger.info("Loading data and model...")
aggregated_2021_2024_90 = pd.read_csv("aggregated_2021_2024_90.csv")
model = tf.keras.models.load_model("player_recommender_model.h5")

# Preprocess data
logger.info("Preprocessing data...")
num_cols = aggregated_2021_2024_90.select_dtypes(include='number')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(num_cols)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

def get_embeddings(model, data):
    embedding_model = tf.keras.models.Sequential(model.layers[:-1])
    embeddings = embedding_model.predict(data)
    return embeddings

logger.info("Generating embeddings...")
embeddings = get_embeddings(model, X_pca)

def fetch_player_image(player_name):
    """Fetch the first image URL from DuckDuckGo image search."""
    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.images(f"{player_name} football player profile pic 360x360", max_results=1))
        return search_results[0]['image'] if search_results else "https://example.com/default.jpg"
    except Exception as e:
        logger.error(f"Error fetching image for {player_name}: {e}")
        return "https://example.com/default.jpg"

player_images = {}

def get_player_image(player_name):
    if player_name not in player_images:
        player_images[player_name] = fetch_player_image(player_name)
    return player_images[player_name]

def find_similar_players(player_name, embeddings, player_data):
    player_index = player_data[player_data['Player'] == player_name].index
    if player_index.empty:
        return []

    player_index = player_index[0]
    similarities = cosine_similarity([embeddings[player_index]], embeddings)[0]
    similar_players_indices = similarities.argsort()[::-1][1:]

    similar_players = player_data.iloc[similar_players_indices][['Player', 'Nation', 'League', 'Squad', 'Age', 'Position']].copy()
    similar_players['similarity_score'] = similarities[similar_players_indices]

    # Fetch images dynamically
    similar_players['image'] = similar_players['Player'].apply(get_player_image)

    return similar_players.to_dict('records')

@app.route('/find_similar_players', methods=['POST'])
def api_find_similar_players():
    start_time = time.time()
    data = request.json
    player_name = data.get('player_name')
    if not player_name:
        return jsonify({"error": "Player name is required"}), 400

    logger.info(f"Finding similar players for {player_name}...")
    similar_players = find_similar_players(player_name, embeddings, aggregated_2021_2024_90)
    logger.info(f"Found {len(similar_players)} similar players in {time.time() - start_time:.2f} seconds.")
    return jsonify(similar_players)

if __name__ == '__main__':
    app.run(port=5007)