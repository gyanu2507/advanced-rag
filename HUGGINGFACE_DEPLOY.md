# 🤗 Deploy on Hugging Face Spaces

Complete guide to deploy your AI Document Q&A app on Hugging Face Spaces (100% FREE).

## 🚀 Quick Deploy Steps

### Step 1: Prepare Your Repository

Your code is already on GitHub at: `https://github.com/gyanu2507/advanced-rag`

### Step 2: Create Hugging Face Space

1. **Go to [Hugging Face Spaces](https://huggingface.co/spaces)**
2. **Click "Create new Space"**
3. **Fill in the details:**
   - **Space name**: `advanced-rag` (or your choice)
   - **SDK**: Select **Streamlit** ⭐
   - **Visibility**: Public (for free hosting)
   - **Hardware**: CPU Basic (free tier)
4. **Click "Create Space"**

### Step 3: Connect GitHub Repository

1. **In your Space settings**, go to "Repository" tab
2. **Click "Import from GitHub"**
3. **Select your repository**: `gyanu2507/advanced-rag`
4. **Select branch**: `main`
5. **Root directory**: Leave empty (or `/` if needed)
6. **Click "Import"**

### Step 4: Configure Environment Variables

1. **Go to Space Settings → Variables and secrets**
2. **Add these variables:**
   ```
   API_URL = https://your-backend-url.railway.app
   ```
   (You'll need to deploy backend separately - see Step 5)

### Step 5: Deploy Backend (Required)

Since Hugging Face Spaces is for frontend, you need to deploy backend separately:

#### Option A: Railway (Recommended - Free Tier)
```bash
npm i -g @railway/cli
railway login
railway init
railway up
```
Get your backend URL and add it to Hugging Face Space variables.

#### Option B: Fly.io (Free Tier)
```bash
fly auth login
fly launch
fly deploy
```

### Step 6: Update app.py for Hugging Face

Hugging Face Spaces may need a small adjustment. Create `app_hf.py` or update `app.py`:

```python
# At the top of app.py, ensure API_URL can be set from environment
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

### Step 7: Create requirements.txt

Make sure `requirements.txt` includes all dependencies (already done ✅)

### Step 8: Deploy!

1. **Hugging Face will auto-detect** your Streamlit app
2. **It will install dependencies** from `requirements.txt`
3. **Your app will be live** at: `https://huggingface.co/spaces/YOUR_USERNAME/advanced-rag`

## 📋 Files Needed for Hugging Face

Your repository already has:
- ✅ `app.py` - Streamlit app
- ✅ `requirements.txt` - Dependencies
- ✅ `.streamlit/config.toml` - Streamlit config

**Optional but recommended:**
- `README.md` - Will show on your Space page
- `app.py` - Main file (must be named `app.py` or `main.py`)

## 🔧 Configuration

### Space Settings:

1. **SDK**: Streamlit
2. **Hardware**: CPU Basic (free) or upgrade if needed
3. **Variables**: Set `API_URL` to your backend URL

### Environment Variables:

In Space Settings → Variables:
```
API_URL=https://your-backend.railway.app
```

## ⚠️ Important Notes

1. **Backend Required**: Hugging Face Spaces only hosts the frontend. You MUST deploy backend separately (Railway/Fly.io)

2. **File Size Limits**: 
   - Free tier: 50GB storage
   - Model downloads count towards storage

3. **Build Time**: First build may take 10-15 minutes (installing dependencies)

4. **Auto-Deploy**: Space auto-updates when you push to GitHub

5. **Custom Domain**: Not available on free tier

## 🎯 Complete Setup Example

### 1. Deploy Backend First (Railway):

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Get your backend URL (e.g., https://backend-production.up.railway.app)
```

### 2. Create Hugging Face Space:

1. Go to https://huggingface.co/spaces
2. Create new Space → Streamlit SDK
3. Import from GitHub: `gyanu2507/advanced-rag`
4. Set variable: `API_URL=https://your-backend.railway.app`
5. Wait for build (10-15 min)
6. Done! 🎉

## 🔗 Your Live URLs

- **Frontend**: `https://huggingface.co/spaces/YOUR_USERNAME/advanced-rag`
- **Backend**: `https://your-backend.railway.app` (or Fly.io URL)

## 🆘 Troubleshooting

### Build Fails:
- Check `requirements.txt` for all dependencies
- Verify Python version compatibility
- Check build logs in Space

### Backend Connection Issues:
- Verify `API_URL` is set correctly
- Check backend is running
- Test backend health: `curl https://your-backend.railway.app/health`

### App Not Loading:
- Check Streamlit logs in Space
- Verify `app.py` is in root directory
- Check for import errors

## 💡 Pro Tips

1. **Monitor Usage**: Check Space settings for resource usage
2. **Auto-Sleep**: Free tier spaces sleep after inactivity (wakes on access)
3. **GitHub Sync**: Push to GitHub → Space auto-updates
4. **Logs**: View logs in Space → Logs tab
5. **Community**: Share your Space URL for others to use!

## 📚 Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Streamlit on Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit)
- [Railway Docs](https://docs.railway.app)

---

**Your app will be live and accessible to everyone for FREE!** 🎉

