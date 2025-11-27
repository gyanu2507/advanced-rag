# 🆓 Free Deployment Options

## Best Free Options for Full-Stack Deployment

### Option 1: Streamlit Cloud (Frontend) + Railway (Backend) ⭐ RECOMMENDED

**Frontend - Streamlit Cloud (100% Free):**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Connect your repo: `gyanu2507/advanced-rag`
4. Main file: `app.py`
5. **Important:** Set environment variable:
   - `API_URL`: Your Railway backend URL
6. Deploy!

**Backend - Railway (Free Tier Available):**
1. Go to [railway.app](https://railway.app)
2. Sign up (free tier available)
3. New Project → Deploy from GitHub
4. Select your repo
5. Railway auto-detects Python
6. Set start command: `python3 -m uvicorn backend:app --host 0.0.0.0 --port $PORT`
7. Get your backend URL and update frontend `API_URL`

**Cost:** $0 (Free tier on Railway, unlimited free on Streamlit Cloud)

---

### Option 2: Fly.io (Full Stack - Free Tier)

**Both Frontend & Backend on Fly.io:**

1. **Install Fly CLI:**
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login:**
   ```bash
   fly auth login
   ```

3. **Deploy Backend:**
   ```bash
   cd /path/to/project
   fly launch --name supernal-backend
   # Select Python, port 8000
   fly deploy
   ```

4. **Deploy Frontend:**
   ```bash
   fly launch --name supernal-frontend
   # Select Python, port 8501
   # Set env: API_URL=https://supernal-backend.fly.dev
   fly deploy
   ```

**Cost:** $0 (Free tier: 3 shared VMs, 3GB storage)

---

### Option 3: Hugging Face Spaces (Frontend Only - Free)

**Deploy Streamlit on Hugging Face:**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create New Space
3. Select "Streamlit" SDK
4. Connect GitHub repo
5. **Note:** You'll need to deploy backend separately (Railway/Fly.io)

**Cost:** $0 (Completely free)

---

### Option 4: Replit (Full Stack - Free Tier)

1. Go to [replit.com](https://replit.com)
2. Import from GitHub
3. Run both backend and frontend
4. Get public URL

**Cost:** $0 (Free tier available)

---

### Option 5: PythonAnywhere (Free Tier)

1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files
3. Configure web app
4. **Limitation:** Free tier has restrictions

**Cost:** $0 (Free tier with limitations)

---

## 🎯 Recommended: Streamlit Cloud + Railway

**Why this combination:**
- ✅ Both have free tiers
- ✅ Streamlit Cloud: Unlimited free hosting
- ✅ Railway: $5 free credit monthly (enough for small apps)
- ✅ Easy setup
- ✅ Auto-deploy from GitHub

### Setup Steps:

1. **Deploy Backend on Railway:**
   ```bash
   npm i -g @railway/cli
   railway login
   railway init
   railway up
   ```
   - Get your backend URL (e.g., `https://backend-production.up.railway.app`)

2. **Deploy Frontend on Streamlit Cloud:**
   - Go to share.streamlit.io
   - Connect GitHub repo
   - Set environment variable: `API_URL=https://your-backend-url.railway.app`
   - Deploy!

3. **Done!** Your app is live for free! 🎉

---

## 📝 Environment Variables

### Frontend (Streamlit Cloud):
- `API_URL`: Your Railway backend URL

### Backend (Railway):
- `PORT`: Auto-set by Railway
- `API_URL`: Your backend URL (optional)

---

## 💡 Tips for Free Deployment

1. **Use Railway's free tier** - $5 credit/month (usually enough)
2. **Streamlit Cloud** - Completely free, no limits
3. **Optimize model loading** - Use smaller models to reduce memory
4. **Enable auto-sleep** - Railway spins down inactive apps (saves credits)
5. **Monitor usage** - Check Railway dashboard for credit usage

---

## 🚀 Quick Deploy Commands

### Railway (Backend):
```bash
railway login
railway init
railway up
```

### Streamlit Cloud:
- Just connect GitHub repo and deploy!

---

**All options are free to start!** Choose based on your needs.

