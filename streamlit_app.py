import streamlit as st
import subprocess
import sys
import time
import os

# Wrapper to run both Backend (FastAPI) and Frontend (Streamlit) in one container
# This is necessary for Streamlit Cloud deployments where you only get one container.

def start_backend():
    """Starts the FastAPI backend using uvicorn in a subprocess."""
    print("Starting FastAPI backend...")
    # Command: uvicorn backend.main:app --host 127.0.0.1 --port 8000
    # We use sys.executable to ensure we use the same python environment
    cmd = [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
    # Run in the current directory (root of repo)
    process = subprocess.Popen(cmd)
    return process

if "backend_process" not in st.session_state:
    # Start the backend only once per session state initialization
    # Note: Streamlit re-runs script on interaction, so we use session_state or singleton
    # However, on cloud, multiple users might share a container, but usually streamlit isolates sessions.
    # A better approach for single-container app is @st.cache_resource
    pass

@st.cache_resource
def launch_backend_server():
    process = start_backend()
    time.sleep(5) # Give it some time to start
    return process

# Launch backend
launch_backend_server()

# Run Frontend Logic
# We can import the app functionality or just execute the file.
# Executing is safer to avoid import issues with relative paths in app.py if it wasn't designed as a module.
try:
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend/app.py")
    with open(frontend_path, encoding='utf-8') as f:
        code = f.read()
        exec(code, globals())
except Exception as e:
    st.error(f"Error loading frontend: {e}")
