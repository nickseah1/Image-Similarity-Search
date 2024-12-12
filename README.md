# Image-Similarity-Search



The KV Rocks Demo Version is an advanced image similarity search application that allows users to upload images and retrieve visually similar images from a dynamically managed dataset. By leveraging FAISS for efficient similarity searches and KvRocksDB for scalable embedding storage, the application combines speed and flexibility, dynamically rebuilding search indices to handle large datasets while maintaining accuracy and consistency.



### Key Components
Data Storage with KvRocksDB:

  Embeddings Storage: Stores high-dimensional vector embeddings of images, facilitating efficient similarity comparisons.

  Metadata Storage: Keeps track of image metadata, such as file paths and unique identifiers.

  Image Hashing: Utilizes SHA256 hashing to detect and prevent duplicate images from being indexed, ensuring each image is uniquely represented.

Embedding Extraction:

  Embedding Extractor: Processes images to generate their corresponding vector embeddings using a pre-trained model (e.g., a convolutional neural network). These embeddings capture the visual features of the images, enabling meaningful similarity comparisons.
  FAISS Indexing:

Hybrind FAISS Search
  
  FAISS Handler: Orchestrates the creation, querying, and dynamic rebuilding of the FAISS index.
    
  Dynamic Indexing: Handles datasets that exceed memory capacity by selectively fetching embeddings from KvRocksDB and rebuilding the FAISS index when needed.
  Cosine Similarity: Uses cosine similarity (via inner product indexing with normalized embeddings) to rank the relevance of images.
  
  Query-Based Index Updates: Ensures rapid search results for frequently queried embeddings by dynamically adapting the FAISS index to include relevant vectors.



Dash Web Interface:

User-Friendly UI: Provides an intuitive interface for users to upload images via drag-and-drop or file selection.
Display Results: Shows the uploaded image alongside the top 5 visually similar images retrieved from the dataset, complete with similarity scores.
Feedback Mechanisms: Informs users about successful uploads, duplicate detections, and any issues encountered during processing.


### Workflow

Image Upload:

Users upload an image through the Dash web interface.
The application validates the image type and size to ensure it's suitable for processing.
Duplicate Detection:

The uploaded image undergoes SHA256 hashing.
The application checks KvRocksDB to determine if the image has already been indexed.
If a duplicate is found, the existing embedding is used for the similarity search, preventing redundant indexing.

Embedding Extraction & Storage:

For new images, the embedding extractor generates a vector representation of the image.
The embedding and corresponding metadata (e.g., image path) are stored in KvRocksDB.
The image hash is also stored to facilitate future duplicate checks.

Hybrid Search and FAISS Indexing

  The FAISS handler manages a dynamic in-memory index for rapid similarity searches:
    Initial FAISS Search:
      Performs an in-memory search for the nearest neighbors of the query embedding.
    Dynamic Rebuild (if needed):
      If insufficient results are found in memory, the application fetches additional embeddings from KvRocksDB and rebuilds the FAISS index dynamically.
    Result Filtering:
      Excludes the query image and ensures diversity in the results.
  Embeddings are normalized before indexing to improve the accuracy of cosine similarity measurements.

Similarity Search:

The application performs a similarity search using the embedding of the uploaded image.
To ensure consistent retrieval of 5 unique similar images, the application fetches additional results (k_plus) and filters out duplicates and the uploaded image itself.
The top 5 similar images, based on cosine similarity scores, are selected for display.
Result Display:

The uploaded image and its 5 most similar counterparts are displayed on the web interface.
Each similar image is accompanied by a similarity score indicating how closely it matches the query image.

### Preventing Duplicate Results

Image Hashing: By generating and storing a unique hash for each image, the application ensures that identical images are not indexed multiple times.

Deduplication Logic: During similarity searches, the application tracks already displayed images to prevent showing the same image more than once, enhancing the relevance and diversity of the results.

### Additional Features

Logging: Comprehensive logging mechanisms record all significant actions and events, aiding in monitoring and debugging.

Utility Scripts: Scripts for flushing the KvRocksDB database and setting up datasets like CIFAR-10 provide flexibility and ease of maintenance.

Scalability: The combination of FAISS and KvRocksDB allows the application to handle large datasets efficiently, ensuring quick response times even as the number of indexed images grows.

### Use Cases

Personal Collections: Users can manage and explore their personal image collections by finding similar photos effortlessly.

E-commerce: Retailers can enhance product searches by allowing customers to find visually similar items.

Research: Academics and developers can utilize the application for experiments involving image retrieval and similarity assessments.

---

## Directory Structure

```plaintext
KV Rocks Demo Version - final/
│
├── data/                             # Dataset directory
│   ├── custom_images/                # Folder for any custom image dataset
│   └── embeddings/                   # Store extracted embeddings (optional for reusability)
│
├── src/                              # Main source code directory
│   ├── __init__.py                   # Empty file to make this a package
│   ├── config.py                     # Configuration variables (e.g., paths, parameters)
│   ├── faiss_handler.py              # FAISS-related operations (indexing, hybrid searching)
│   ├── kvrocksdb_handler.py          # KvRocksDB-related operations (storage, retrieval)
│   ├── embedding_extractor.py        # Model to generate image embeddings
│   ├── data_loader.py                # Data loading utilities (for CIFAR or custom dataset)
│   └── visualize_results.py          # Utilities to display query and search results
│
├── scripts/                          # Additional utility scripts
│   ├── flush_kvrocksdb.py            # Script to flush KvRocksDB database
│   └── setup_cifar10.py               # (Optional) Script to download and prepare CIFAR-10 dataset
│
├── notebooks/                        # Jupyter notebooks for experimentation
│   └── demo.ipynb                    # Interactive demo notebook
│
├── app.py                            # Dash application script
├── main.py                           # Main script to run the demo
│
├── requirements.txt                  # Python dependencies
│
├── README.md                         # Project description and instructions
│
└── .gitignore                        # Ignore unnecessary files (e.g., KvRocksDB database, logs)

### This project showcases how to combine FAISS and RocksDB for a hybrid image similarity search pipeline. It leverages image embeddings generated by pre-trained models to perform efficient similarity searches and manage the corresponding metadata in RocksDB.

---

### FAISS and KvRocksDB Integration Demo

This project demonstrates the integration of [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search with [KvRocksDB](https://github.com/apache/incubator-kvrocks) for key-value storage.

## Prerequisites

Ensure you have the following installed on your system:
- Python (>= 3.8)
- Docker and Docker Compose
- Redis client library (e.g., `redis-py`)
- FAISS library
- Required Python dependencies (see `requirements.txt`)

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

### 2. Install Python Dependencies

pip install -r requirements.txt


### 3: Pull the Docker Image
Run the following command to pull the KvRocksDB Docker image:
```bash
docker pull kvrocks/kvrocks:latest
docker run --name kvrocks -p 6666:6666 -d kvrocks/kvrocks:latest

Connecto to KvRocksDB using Redis client

redis-cli -p 6666


### 4. Run Application (Ensure the files are setup as per the directory shown above and there is an image named query.jpg in /data/custom_images.


python main.py
python app.py
