# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

# Location of demo models
export MODEL_NAME="seal_detector"
export MODEL_DATA="$(pwd)/saved_model/"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$MODEL_DATA:/models/${MODEL_NAME}" \
    -e MODEL_NAME=${MODEL_NAME} \
    tensorflow/serving
