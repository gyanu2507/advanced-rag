#!/bin/bash

# Quick Deployment Script
echo "🚀 AI Document Q&A - Deployment Helper"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit - Ready for deployment"
    echo "✅ Git initialized"
    echo ""
fi

echo "Choose deployment platform:"
echo "1. Render.com (Recommended - Free tier available)"
echo "2. Railway (Easy deployment)"
echo "3. Streamlit Cloud (Frontend only)"
echo "4. Docker (Local/Cloud)"
echo "5. Heroku"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "📋 Render.com Deployment Steps:"
        echo "1. Push to GitHub:"
        echo "   git remote add origin <your-repo-url>"
        echo "   git push -u origin main"
        echo ""
        echo "2. Go to https://render.com and:"
        echo "   - Create account"
        echo "   - New Web Service"
        echo "   - Connect GitHub repo"
        echo "   - Use render.yaml (auto-detected)"
        echo ""
        echo "3. Set environment variables:"
        echo "   API_URL=https://your-backend.onrender.com"
        ;;
    2)
        echo ""
        echo "📋 Railway Deployment Steps:"
        echo "1. Install Railway CLI:"
        echo "   npm i -g @railway/cli"
        echo ""
        echo "2. Deploy:"
        echo "   railway login"
        echo "   railway init"
        echo "   railway up"
        ;;
    3)
        echo ""
        echo "📋 Streamlit Cloud Deployment:"
        echo "1. Push to GitHub"
        echo "2. Go to https://share.streamlit.io"
        echo "3. Deploy app.py"
        echo "4. Note: Deploy backend separately"
        ;;
    4)
        echo ""
        echo "📋 Docker Deployment:"
        echo "1. Build: docker build -t supernal-app ."
        echo "2. Run: docker-compose up -d"
        echo "3. Or push to Docker Hub and deploy on cloud"
        ;;
    5)
        echo ""
        echo "📋 Heroku Deployment:"
        echo "1. Install Heroku CLI"
        echo "2. heroku create supernal-app"
        echo "3. git push heroku main"
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo ""
echo "📖 For detailed instructions, see DEPLOYMENT.md"
echo ""

