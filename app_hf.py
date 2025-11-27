"""
Streamlit app optimized for Hugging Face Spaces deployment.
This is a wrapper that ensures proper configuration for HF Spaces.
"""
import os
import sys

# Ensure API_URL is set from environment
os.environ.setdefault("API_URL", os.getenv("API_URL", "http://localhost:8000"))

# Import the main app
# Hugging Face will look for app.py, so we'll use this as a reference
# In HF Spaces, just use app.py directly

if __name__ == "__main__":
    # This file is for reference
    # Hugging Face Spaces uses app.py directly
    print("For Hugging Face Spaces, use app.py directly")
    print("Make sure API_URL environment variable is set in Space settings")

