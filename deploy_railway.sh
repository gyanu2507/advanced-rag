#!/bin/bash

# Railway Deployment Script
# Quick deploy your app to Railway

echo "🚂 Railway Deployment Script"
echo "============================"
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found!"
    echo ""
    echo "📦 Installing Railway CLI..."
    echo ""
    echo "Choose installation method:"
    echo "1. npm (Node.js required)"
    echo "2. Homebrew (Mac)"
    echo "3. curl script"
    echo ""
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            npm i -g @railway/cli
            ;;
        2)
            brew install railway
            ;;
        3)
            curl -fsSL https://railway.app/install.sh | sh
            ;;
        *)
            echo "Invalid choice. Please install Railway CLI manually:"
            echo "  npm i -g @railway/cli"
            echo "  or visit: https://docs.railway.app/develop/cli"
            exit 1
            ;;
    esac
fi

echo ""
echo "✅ Railway CLI found!"
echo ""

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Logging in to Railway..."
    railway login
else
    echo "✅ Already logged in to Railway"
    USER=$(railway whoami 2>/dev/null | head -n 1)
    echo "   User: $USER"
fi

echo ""
echo "📦 Deploying backend..."
echo ""

# Check if project is linked
if [ ! -f ".railway" ]; then
    echo "🔗 Initializing Railway project..."
    railway init
else
    echo "✅ Project already linked"
fi

echo ""
echo "🚀 Starting deployment..."
echo ""

# Deploy
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Get your backend URL:"
echo "   railway domain"
echo ""
echo "2. Deploy frontend:"
echo "   Option A: Streamlit Cloud (free)"
echo "   - Go to: https://share.streamlit.io"
echo "   - Connect GitHub repo"
echo "   - Set API_URL to your Railway backend URL"
echo ""
echo "   Option B: Railway (same project)"
echo "   - railway service create frontend"
echo "   - railway service use frontend"
echo "   - railway variables set API_URL=https://your-backend.up.railway.app"
echo "   - railway up"
echo ""
echo "3. View logs:"
echo "   railway logs"
echo ""
echo "🎉 Done!"

