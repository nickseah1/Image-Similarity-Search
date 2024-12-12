import sys
import os
import base64
import numpy as np
import logging
import tempfile
import hashlib  # Added for hash computation

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from kvrocksdb_handler import KVRocksDBHandler
from embedding_extractor import EmbeddingExtractor
from faiss_handler import FaissHandlerWithKVRocks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
DIMENSION = 2048
kv_handler = KVRocksDBHandler()
faiss_handler = FaissHandlerWithKVRocks(DIMENSION, kv_handler)
extractor = EmbeddingExtractor()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    html.H1("Image Similarity Search", style={"text-align": "center", "margin-top": "20px"}),
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
        style={
            'width': '80%',
            'height': '80px',
            'lineHeight': '80px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': '20px auto'
        },
        multiple=False
    ),
    html.Div(id='output-image-upload', style={"text-align": "center", "margin": "10px"}),
    html.Div(id='similar-images', style={"text-align": "center", "margin": "20px"})
])

def encode_image(image_path):
    """Encode an image file to a Base64 string."""
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return ""

    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')

        ext = os.path.splitext(image_path)[1].lower()
        mime = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif'
        }.get(ext, 'application/octet-stream')

        return f'data:{mime};base64,{encoded}'
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

def compute_image_hash(image_path):
    """Compute MD5 hash of the image for duplicate detection."""
    hash_md5 = hashlib.md5()
    try:
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {image_path}: {e}")
        return None

def process_uploaded_image(contents):
    """Convert uploaded image to an embedding and compute its hash."""
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(decoded)
            temp_image_path = tmp.name
        embedding = extractor.extract(temp_image_path)
        if embedding is None or len(embedding) != DIMENSION:
            raise ValueError("Invalid embedding or dimension mismatch.")
        
        # Compute hash of the uploaded image
        image_hash = compute_image_hash(temp_image_path)
        if image_hash is None:
            raise ValueError("Failed to compute image hash.")
        
        return embedding, temp_image_path, image_hash
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        return None, None, None

@app.callback(
    [Output('output-image-upload', 'children'),
     Output('similar-images', 'children')],
    [Input('upload-image', 'contents')]
)
def update_output(contents):
    """Handle the uploaded image, perform similarity search, and display results."""
    if contents is not None:
        logger.info("Processing uploaded image...")

        # Process uploaded image
        query_embedding, temp_image_path, image_hash = process_uploaded_image(contents)
        if query_embedding is None:
            return html.Div("Failed to process the uploaded image.", style={"color": "red"}), None

        # Check if the uploaded image already exists in the database
        existing_key = kv_handler.retrieve_metadata(f"hash_{image_hash}")
        if existing_key:
            logger.info(f"Uploaded image already exists in the database with key: {existing_key}")
            exclude_keys = {existing_key}
        else:
            logger.info("Uploaded image is new and not present in the database.")
            exclude_keys = set()

        # Perform hybrid similarity search with k=6 to account for possible duplicate
        logger.info("Performing hybrid search...")
        k = 6  # Request one extra to filter out duplicates if necessary
        similar_keys, similar_distances = faiss_handler.hybrid_search(query_embedding, k=k)

        # Retrieve metadata for the results
        similar_images = []
        seen_image_paths = set()
        for similar_key, dist in zip(similar_keys, similar_distances):
            if similar_key in exclude_keys:
                logger.info(f"Excluding image with key {similar_key} as it is the uploaded image.")
                continue  # Exclude the uploaded image if it exists in the database

            metadata_key = f"metadata_{similar_key}"
            img_path = kv_handler.retrieve_metadata(metadata_key)
            if img_path and os.path.exists(img_path):
                # Compute hash of the similar image
                similar_image_hash = compute_image_hash(img_path)
                if similar_image_hash == image_hash:
                    logger.info(f"Excluding image {img_path} as it has the same hash as the uploaded image.")
                    continue  # Exclude images with the same hash
                # Encode image
                encoded_img = encode_image(img_path)
                if encoded_img and img_path not in seen_image_paths:
                    similar_images.append(encoded_img)
                    seen_image_paths.add(img_path)
                if len(similar_images) >= 5:
                    break  # Only display top 5 similar images

        # Display uploaded image with label
        uploaded_image_display = html.Div([
            html.H4("Uploaded Image", style={"text-align": "center", "margin-bottom": "10px"}),
            html.Img(src=contents, style={"height": "300px", "margin-bottom": "20px"})
        ])

        # Display similar images with label
        similar_images_display = html.Div([
            html.H4("Similar Images", style={"text-align": "center", "margin-bottom": "10px"}),
            html.Div([
                html.Div([
                    html.Img(src=img_src, style={"height": "200px", "margin": "10px"})
                ]) for img_src in similar_images
            ], style={"display": "flex", "flex-wrap": "wrap", "justify-content": "center"})
        ]) if similar_images else html.Div("No similar images found.", style={"color": "red"})

        return uploaded_image_display, similar_images_display

    return None, None

# Run the app
if __name__ == '__main__':
    logger.info("Starting Dash app...")
    app.run_server(debug=True)
