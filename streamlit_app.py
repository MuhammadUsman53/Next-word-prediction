import streamlit as st
import subprocess
import sys
import time
import os
import requests

# --- Backend Launcher ---

# Function to start the backend with caching to run only once
@st.cache_resource
def start_backend():
    """Starts the FastAPI backend using uvicorn in a subprocess."""
    print("Starting FastAPI backend...")
    
    # Ensure current working directory is correct
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    # Construct command
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    
    # Start process
    # Use cwd=cwd to ensure relative imports inside backend work if needed, 
    # but since main.py is in backend/, we need root as cwd for "backend.main" to import
    # So cwd should be the root of the project (where this script is)
    process = subprocess.Popen(cmd, cwd=cwd)
    
    # Wait a bit for startup
    time.sleep(5) 
    
    return process

# Launch backend
backend_process = start_backend()


# --- Frontend UI ---

st.title("Next Word Prediction System")
st.write("Enter a sequence of words, and the model will predict the next word.")
st.caption("Backend Status: Running on port 8000")

input_text = st.text_input("Enter text:", "The Pakistani rupee")

if st.button("Predict Next Word"):
    if input_text:
        try:
            # We use localhost because backend is in the same container/machine
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": input_text},
                timeout=10 # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                prediction = response.json().get("prediction", "")
                st.success(f"Predicted next word: **{prediction}**")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Error: Could not connect to backend server. Make sure it's running.")
        except requests.exceptions.Timeout:
            st.error("Error: Connection timed out.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter some text first.")
