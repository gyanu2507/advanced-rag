#!/bin/bash

# Startup script for AI Document Q&A System

echo "🚀 Starting AI Document Q&A System..."
echo ""

# Check if .env exists (optional - not required for free mode)
if [ ! -f .env ]; then
    echo "ℹ️  No .env file found - running in FREE mode (no API key needed)"
    echo "   To use Hugging Face API (optional, faster), create .env with:"
    echo "   HUGGINGFACE_API_TOKEN=your_token_here"
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Start backend in background
echo "🔧 Starting backend server..."
python backend.py &
BACKEND_PID=$!
sleep 3

# Start frontend
echo "🎨 Starting frontend..."
echo "Open your browser to http://localhost:8501"
echo ""
streamlit run app.py

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT

