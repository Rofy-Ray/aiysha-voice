import logging
import os
import time
import openai
import requests
from dotenv import load_dotenv
from functools import lru_cache
from google.cloud import storage
from langchain_chroma import Chroma
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.auth import default, transport
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

ASR_ENDPOINT = os.getenv("ASR_ENDPOINT")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT")
MAAS_ENDPOINT = os.getenv("MAAS_ENDPOINT")
PROJECT_NUMBER = os.getenv("PROJECT_NUMBER")
BUCKET_NAME = os.getenv("BUCKET_NAME")

CHROMA_PATH = "database/"
DB_DIR = "chroma_db"
LOCK_FILE = os.path.join(DB_DIR, "lock")

bucket = storage.Client().bucket(BUCKET_NAME)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

def acquire_lock():
    if os.path.exists(LOCK_FILE):
        logger.warning("Lock file already exists. Waiting for lock to be released.")
        while os.path.exists(LOCK_FILE):
            time.sleep(1)
    with open(LOCK_FILE, "w") as f:
        f.write("locked")

def release_lock():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

@lru_cache(maxsize=1)
def get_embedding_function():
    try:
        logger.debug("Initializing FastEmbedEmbeddings")
        embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        logger.debug("FastEmbedEmbeddings initialized")
        
        logger.debug("Testing embed_query")
        test_embed = embedding_function.embed_query("test")
        logger.info(f"Embedding function initialized successfully. Test embedding shape: {len(test_embed)}")
        
        return embedding_function
    except Exception as e:
        logger.error(f"Error initializing embedding function: {str(e)}", exc_info=True)
        raise
    
def download_db_from_gcs(local_dir):
    blobs = bucket.list_blobs(prefix=CHROMA_PATH)
    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name[len(CHROMA_PATH):])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
    logger.info(f"Downloaded Chroma DB from GCS to {local_dir}")

@lru_cache(maxsize=1)
def get_vector_store():
    acquire_lock()
    logger.debug("Entering get_vector_store")
    try:
        logger.debug("Getting embedding function")
        embedding_function = get_embedding_function()
        logger.debug("Embedding function retrieved")

        logger.debug(f"Checking for Chroma database in GCS at {CHROMA_PATH}chroma.sqlite3")
        if not bucket.blob(f"{CHROMA_PATH}chroma.sqlite3").exists():
            logger.warning(f"Chroma database not found in GCS at {CHROMA_PATH}. Creating a new one.")
            logger.debug(f"Initializing Chroma with persist_directory={DB_DIR}")
            db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
        else:
            logger.debug("Downloading existing database from GCS")
            download_db_from_gcs(DB_DIR)
            logger.debug(f"Initializing Chroma with persist_directory={DB_DIR}")
            db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
            
        db_file_path = os.path.join(DB_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file_path):
            logger.error(f"Database file not found at {db_file_path}")
            raise FileNotFoundError(f"Database file not found at {db_file_path}")
        
        logger.debug("Vector store initialized successfully")
        return db
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}", exc_info=True)
        return None
    finally:
        release_lock()

def query_vector_store(query: str, db, k: int = 3):
    try:
        results = db.similarity_search_with_score(query, k=k)
        if not results:
            logger.warning("No results found for the given query.")
            return ""
        context = "\n\n".join([doc.page_content for doc, score in results])
        return context
    except Exception as e:
        logger.error(f"Error in query_vector_store: {str(e)}")
        raise

vector_store = get_vector_store()

def transcribe_audio(audio_file):
    try:
        files = {'file': ('audio.wav', audio_file, 'audio/wav')}
        response = requests.post(ASR_ENDPOINT, files=files)
        response.raise_for_status()
        result = response.json()
        return result['text']
    except requests.RequestException as e:
        logger.error(f"Error in ASR request: {str(e)}")
        return None

def text_to_speech(text):
    try:
        payload = {"text": text}
        response = requests.post(TTS_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['audio_url']
    except requests.RequestException as e:
        logger.error(f"Error in TTS request: {str(e)}")
        return None

def get_openai_client():
    credentials, _ = default()
    auth_request = transport.requests.Request()
    credentials.refresh(auth_request)
    return openai.OpenAI(
        base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/openapi/chat/completions?",
        api_key=credentials.token
    )

def get_text_response(message: str, context: str):
    client = get_openai_client()
    
    system_message = """
    You are a makeup artist and beauty advisor named Aiysha. You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
    You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products. 
    Please respond with complete sentences and keep your responses under 280 characters.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {message}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="meta/llama3-405b-instruct-maas",
            messages=messages,
            max_tokens=4096,
        )
        
        generated_content = response.choices[0].message.content if response.choices else None
        
        if generated_content:
            return generated_content
        else:
            logger.warning("Empty response from text model")
            return "I'm sorry, I couldn't generate a response. Could you please try rephrasing your question?"
    
    except Exception as e:
        logger.error(f"Error in text response generation: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

@app.route('/aiyshaspeech', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        transcribed_text = transcribe_audio(file)
        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio"}), 500
        
        if vector_store is not None:
            try:
                context = query_vector_store(transcribed_text, vector_store)
            except Exception as e:
                logger.error(f"Error querying vector store: {str(e)}")
                context = "Unable to retrieve context."
        else:
            context = "Vector store is not available."
        
        try:
            llm_response = get_text_response(transcribed_text, context)
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return jsonify({"error": "No response from LLM"}), 500
        
        audio_url = text_to_speech(llm_response)
        if not audio_url:
            return jsonify({"error": "Failed to convert text to speech"}), 500
        
        return jsonify({"audio_url": audio_url})
    
    except Exception as e:
        logger.error(f"Unexpected error in process_audio: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))