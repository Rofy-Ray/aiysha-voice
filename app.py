import os
import time
import json
import uuid
import openai
import logging
import requests
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
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
CONVERSATION_TRACK_BLOB = os.getenv("CONVERSATION_TRACK_BLOB")

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
        embedding_function = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")
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

def generate_conversation_id():
    """Generate a unique conversation ID."""
    return str(uuid.uuid4())

def save_chat_history(conversation_id: str, messages: list):
    """Save chat history to GCS using conversation ID."""
    try:
        blob = bucket.blob(f"history/chats/{conversation_id}.json")
        blob.upload_from_string(json.dumps(messages, indent=4))
        logger.info(f"Chat history saved for conversation ID: {conversation_id}")
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")
        raise

def load_chat_history(conversation_id: str) -> list:
    """Load chat history from GCS using conversation ID."""
    try:
        blob = bucket.blob(f"history/chats/{conversation_id}.json")
        if blob.exists():
            return json.loads(blob.download_as_string())
        return []
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        return []
    
def update_conversation_count():
    """Update conversation count in GCS."""
    blob = bucket.blob(CONVERSATION_TRACK_BLOB)
    
    try:
        if not blob.exists():
            blob.upload_from_string("queries: 0\nresponses: 0")
            
        content = blob.download_as_text()
        lines = content.split('\n')
        
        queries = int(lines[0].split(': ')[1])
        responses = int(lines[1].split(': ')[1])
        
        queries += 1
        responses += 1
        
        new_content = f"queries: {queries}\nresponses: {responses}"
        blob.upload_from_string(new_content)
        logger.info("Conversation count updated successfully")
    except Exception as e:
        logger.error(f"Error updating conversation count: {str(e)}")
        
def preprocess_audio(file):
    """Convert audio to mono if needed and return temporary file path."""
    try:
        with NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sound = AudioSegment.from_file(file)
            if sound.channels != 1:
                sound = sound.set_channels(1)
            sound.export(temp_file.name, format="wav")
            return temp_file.name
    except Exception as e:
        logger.error(f"Error preprocessing audio: {str(e)}")
        raise
    
def transcribe_audio(audio_path):
    """Transcribe audio file to text."""
    try:
        with open(audio_path, 'rb') as audio_file:
            files = {'file': ('audio.wav', audio_file, 'audio/wav')}
            response = requests.post(ASR_ENDPOINT, files=files)
            response.raise_for_status()
            result = response.json()
            return result['text']
    except requests.RequestException as e:
        logger.error(f"Error in ASR request: {str(e)}")
        raise
    finally:
        try:
            os.unlink(audio_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")
            
def upload_audio_to_gcs(audio_bytes):
    file_name = f"uploads/audio/{uuid.uuid4()}.wav"
    blob = bucket.blob(file_name)
    blob.upload_from_string(audio_bytes, content_type='audio/wav')
    return blob.public_url

def text_to_speech(text):
    """Convert text to speech and return both text and audio URL."""
    try:
        payload = {"text": text}
        response = requests.post(TTS_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        return {
            "audio_url": result['audio_url']
        }
    except requests.RequestException as e:
        logger.error(f"Error in TTS request: {str(e)}")
        raise

def get_openai_client():
    credentials, _ = default()
    auth_request = transport.requests.Request()
    credentials.refresh(auth_request)
    return openai.OpenAI(
        base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{PROJECT_NUMBER}/locations/us-central1/endpoints/openapi/chat/completions?",
        api_key=credentials.token
    )

def get_text_response(message: str, context: str, chat_history: list):
    client = get_openai_client()
        
    system_message = """
    You are a makeup artist and beauty advisor named Aiysha. You apply cosmetics on clients to enhance features, create looks and styles according to the latest trends in beauty and fashion. 
    You offer advice about skincare routines, know how to work with different textures of skin tone, and are able to use both traditional methods and new techniques for applying products. 
    Please respond with complete sentences and keep your responses under 280 characters.
    """
    
    messages = [{"role": "system", "content": system_message}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {message}"})
    
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

@app.route('/aiyshavoice', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        conversation_id = request.form.get('conversation_id')
        is_new_conversation = False
        
        if not conversation_id:
            conversation_id = generate_conversation_id()
            is_new_conversation = True
            logger.info(f"New conversation started with ID: {conversation_id}")
        
        processed_audio_path = preprocess_audio(file)
        
        with open(processed_audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        user_audio_url = upload_audio_to_gcs(audio_bytes)
        
        transcribed_text = transcribe_audio(processed_audio_path)
        
        if not transcribed_text:
            return jsonify({"error": "Failed to transcribe audio"}), 500
        
        context = ""
        if vector_store is not None:
            try:
                context = query_vector_store(transcribed_text, vector_store)
            except Exception as e:
                logger.error(f"Error querying vector store: {str(e)}")
                context = "Unable to retrieve context."
        
        chat_history = load_chat_history(conversation_id) if not is_new_conversation else []
        
        try:
            llm_response = get_text_response(transcribed_text, context, chat_history)
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}")
            return jsonify({"error": "No response from LLM"}), 500
        
        tts_result = text_to_speech(llm_response)
        
        chat_history.extend([
            {"role": "user", "content": transcribed_text, "audio": user_audio_url, "image": None},
            {"role": "assistant", "content": llm_response, "audio": tts_result["audio_url"] if tts_result["audio_url"] else None, "image": None}
        ])
        
        save_chat_history(conversation_id, chat_history)
        
        update_conversation_count()
        
        return jsonify({
            "conversation_id": conversation_id,
            "text": llm_response,
            "audio_url": tts_result["audio_url"]
        })
    
    except Exception as e:
        logger.error(f"Unexpected error in processing audio: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))