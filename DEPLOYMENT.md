# Deployment Guide for AI Document Q&A App

This guide covers multiple deployment options for the RAG-powered Document Q&A application.

## 🚀 Quick Deploy Options

### Option 1: Streamlit Cloud (Easiest - Frontend Only)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `app.py`
   - Click "Deploy"

3. **Note:** You'll need to deploy backend separately (see Option 2 or 3)

---

### Option 2: Render (Full Stack - Recommended)

#### Backend Deployment:

1. **Create account:** [render.com](https://render.com)

2. **Create New Web Service:**
   - Connect your GitHub repository
   - Name: `supernal-backend`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python3 -m uvicorn backend:app --host 0.0.0.0 --port $PORT`
   - Add Environment Variable:
     - `API_URL`: `https://supernal-backend.onrender.com`

3. **Frontend Deployment:**
   - Create another Web Service
   - Name: `supernal-frontend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Add Environment Variable:
     - `API_URL`: `https://supernal-backend.onrender.com` (your backend URL)

4. **Or use render.yaml:**
   - Push `render.yaml` to your repo
   - Render will auto-detect and deploy both services

---

### Option 3: Railway (Full Stack)

1. **Install Railway CLI:**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Deploy:**
   ```bash
   railway init
   railway up
   ```

3. **Set Environment Variables:**
   - `API_URL`: Your backend URL
   - Railway will auto-detect Python and install dependencies

---

### Option 4: Docker Deployment

#### Local Docker:

```bash
# Build and run with docker-compose
docker-compose up -d

# Access:
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

#### Docker on Cloud:

1. **Build image:**
   ```bash
   docker build -t supernal-app .
   ```

2. **Push to Docker Hub:**
   ```bash
   docker tag supernal-app yourusername/supernal-app
   docker push yourusername/supernal-app
   ```

3. **Deploy on:**
   - **Fly.io:** `flyctl launch`
   - **DigitalOcean App Platform:** Use Dockerfile
   - **AWS ECS/Fargate:** Use Docker image
   - **Google Cloud Run:** `gcloud run deploy`

---

### Option 5: Heroku

1. **Install Heroku CLI**

2. **Deploy:**
   ```bash
   heroku create supernal-app
   heroku buildpacks:add heroku/python
   git push heroku main
   ```

3. **Set Config Vars:**
   ```bash
   heroku config:set API_URL=https://your-app.herokuapp.com
   ```

---

## 📋 Pre-Deployment Checklist

- [ ] Update `API_URL` in `app.py` to production URL
- [ ] Test all features locally
- [ ] Ensure database migrations are handled
- [ ] Set up environment variables
- [ ] Configure CORS if needed
- [ ] Test file uploads
- [ ] Verify AI models load correctly

## 🔧 Environment Variables

Create a `.env` file or set in deployment platform:

```env
API_URL=http://localhost:8000  # Update to production URL
```

## 📝 Notes

- **Database:** SQLite file will be created automatically
- **Vector Store:** ChromaDB data persists in `chroma_db/` directory
- **Model Loading:** First request may be slow as models download
- **File Size Limits:** Check platform limits (usually 100MB-500MB)

## 🆘 Troubleshooting

- **Backend not connecting:** Check `API_URL` environment variable
- **Models not loading:** Ensure sufficient memory (2GB+ recommended)
- **File upload fails:** Check file size limits and timeout settings
- **Database errors:** Ensure write permissions for database file

## 🌐 Production URLs

After deployment, update:
- Frontend `API_URL` to point to backend URL
- CORS settings in `backend.py` if needed
- Database path if using cloud storage

---

**Recommended:** Start with **Render** or **Railway** for easiest full-stack deployment!

