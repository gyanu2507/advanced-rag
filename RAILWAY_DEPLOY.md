# 🚂 Railway Deployment Guide

Complete guide to deploy your AI Document Q&A app on Railway (100% FREE tier available).

## 🎯 What You'll Deploy

- **Backend**: FastAPI server (required)
- **Frontend**: Streamlit app (optional, can use Streamlit Cloud instead)

## 🚀 Quick Deploy (5 Minutes)

### Method 1: Railway CLI (Recommended)

#### Step 1: Install Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Or using Homebrew (Mac)
brew install railway

# Or using curl
curl -fsSL https://railway.app/install.sh | sh
```

#### Step 2: Login to Railway

```bash
railway login
```

This will open your browser for authentication.

#### Step 3: Deploy Backend

```bash
# Navigate to your project
cd /Users/gyanu/Desktop/supernal

# Initialize Railway project
railway init

# Link to existing project (if you have one) or create new
# Select: Create new project

# Deploy backend
railway up
```

Railway will:
- Auto-detect Python
- Install dependencies from `requirements.txt`
- Start the backend server

#### Step 4: Get Your Backend URL

```bash
# Get the deployment URL
railway domain
```

Or check Railway dashboard → Your service → Settings → Domains

Your backend URL will be: `https://your-project-name.up.railway.app`

#### Step 5: Deploy Frontend (Optional)

**Option A: Deploy on Railway (Same Project)**

```bash
# Create a new service for frontend
railway service create frontend

# Set the service
railway service use frontend

# Set environment variable
railway variables set API_URL=https://your-backend.up.railway.app

# Deploy
railway up --service frontend
```

**Option B: Deploy on Streamlit Cloud (Recommended - Free)**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub repo
3. Set `API_URL` to your Railway backend URL
4. Deploy!

---

### Method 2: Railway Dashboard (Web UI)

#### Step 1: Create Project

1. Go to [railway.app](https://railway.app)
2. Sign up/Login (free tier available)
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Connect your repository: `gyanu2507/advanced-rag`

#### Step 2: Configure Backend Service

1. Railway auto-detects your project
2. Click "Add Service" → "GitHub Repo"
3. Select your repo
4. Railway will auto-detect it's a Python project

#### Step 3: Configure Backend Settings

1. Go to Service Settings → Variables
2. Add if needed:
   ```
   PORT=8000
   ```
   (Railway auto-sets PORT, but you can specify)

3. Go to Settings → Deploy
4. Set **Start Command**:
   ```
   python3 -m uvicorn backend:app --host 0.0.0.0 --port $PORT
   ```

5. Set **Root Directory**: (leave empty, or `/`)

#### Step 4: Deploy

1. Railway will automatically:
   - Install dependencies from `requirements.txt`
   - Run the start command
   - Generate a public URL

2. Get your backend URL:
   - Go to Settings → Domains
   - Copy the generated URL (e.g., `https://backend-production.up.railway.app`)

#### Step 5: Deploy Frontend (Optional)

**Option A: Add Frontend Service on Railway**

1. In same project, click "Add Service"
2. Select "GitHub Repo" → Your repo
3. Configure:
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
   - **Variables**: 
     ```
     API_URL=https://your-backend.up.railway.app
     PORT=8501
     ```

**Option B: Use Streamlit Cloud (Free)**

- Deploy frontend separately on Streamlit Cloud
- Set `API_URL` to your Railway backend URL

---

## 📋 Configuration Files

### Backend Configuration

Railway uses `railway.json` (already in your repo):

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python3 -m uvicorn backend:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Environment Variables

Set in Railway Dashboard → Variables:

**Backend:**
```
PORT=8000  # Auto-set by Railway
```

**Frontend (if deploying on Railway):**
```
API_URL=https://your-backend.up.railway.app
PORT=8501
```

---

## 🔧 Railway CLI Commands

```bash
# Login
railway login

# Create new project
railway init

# Link to existing project
railway link

# Deploy
railway up

# View logs
railway logs

# Open dashboard
railway open

# Get deployment URL
railway domain

# Set environment variables
railway variables set KEY=value

# View variables
railway variables

# Create new service
railway service create service-name

# Switch service
railway service use service-name
```

---

## 🎯 Recommended Setup

### Backend on Railway + Frontend on Streamlit Cloud

**Why this combination:**
- ✅ Both free tiers
- ✅ Easy to manage
- ✅ Auto-deploy from GitHub
- ✅ Better separation of concerns

**Steps:**

1. **Deploy Backend on Railway:**
   ```bash
   railway login
   railway init
   railway up
   ```
   Get URL: `https://backend-production.up.railway.app`

2. **Deploy Frontend on Streamlit Cloud:**
   - Go to share.streamlit.io
   - Connect GitHub: `gyanu2507/advanced-rag`
   - Set env: `API_URL=https://backend-production.up.railway.app`
   - Deploy!

3. **Done!** 🎉

---

## 💰 Railway Free Tier

**What's included:**
- $5 free credit per month
- 500 hours of usage
- 5GB storage
- 100GB bandwidth

**For small apps:** Usually enough for free usage!

**Monitor usage:** Dashboard → Usage tab

---

## 🐛 Troubleshooting

### Backend Not Starting

**Check logs:**
```bash
railway logs
```

**Common issues:**
1. **Port binding error:**
   - Ensure using `$PORT` environment variable
   - Backend should listen on `0.0.0.0:$PORT`

2. **Dependencies not installing:**
   - Check `requirements.txt` is correct
   - Verify Python version compatibility

3. **Module not found:**
   - Ensure all imports are in `requirements.txt`
   - Check file paths are correct

### Frontend Can't Connect to Backend

1. **Check backend URL:**
   ```bash
   railway domain
   ```

2. **Test backend health:**
   ```bash
   curl https://your-backend.up.railway.app/health
   ```

3. **Verify CORS:**
   - Backend should allow requests from frontend domain
   - Check `backend.py` CORS settings

### Build Fails

1. **Check build logs:**
   ```bash
   railway logs --build
   ```

2. **Common fixes:**
   - Update `requirements.txt` with exact versions
   - Check Python version (Railway uses Python 3.11 by default)
   - Verify all file paths exist

---

## 📊 Monitoring

### View Logs

```bash
# Real-time logs
railway logs --follow

# Specific service
railway logs --service backend
```

### View Metrics

- Go to Railway Dashboard
- Click on your service
- View: CPU, Memory, Network usage

### Set Up Alerts

- Dashboard → Service → Settings → Alerts
- Get notified of crashes or high usage

---

## 🔄 Auto-Deploy from GitHub

Railway automatically deploys when you push to GitHub:

1. **Connect GitHub repo** (done during setup)
2. **Push to main branch:**
   ```bash
   git push origin main
   ```
3. **Railway auto-deploys!** 🚀

### Configure Branch

- Settings → Source → Branch
- Default: `main`
- Change if needed

---

## 🎉 Your Live URLs

After deployment:

- **Backend**: `https://your-project-name.up.railway.app`
- **Frontend**: 
  - If on Railway: `https://frontend-production.up.railway.app`
  - If on Streamlit Cloud: `https://your-app.streamlit.app`

---

## 📚 Additional Resources

- [Railway Docs](https://docs.railway.app)
- [Railway CLI Reference](https://docs.railway.app/develop/cli)
- [Pricing](https://railway.app/pricing)

---

## ✅ Quick Checklist

- [ ] Install Railway CLI
- [ ] Login to Railway
- [ ] Deploy backend (`railway up`)
- [ ] Get backend URL
- [ ] Deploy frontend (Railway or Streamlit Cloud)
- [ ] Set `API_URL` environment variable
- [ ] Test deployment
- [ ] Share your app! 🎉

---

**Your app is now live on Railway!** 🚂✨

