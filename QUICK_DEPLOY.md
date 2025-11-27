# 🚀 Quick Deployment Guide

## Fastest Way: Render.com (Recommended)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Ready for deployment"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

1. **Go to [render.com](https://render.com)** and sign up
2. **Create New Web Service** → Connect GitHub repo
3. **Backend Service:**
   - Name: `supernal-backend`
   - Build: `pip install -r requirements.txt`
   - Start: `python3 -m uvicorn backend:app --host 0.0.0.0 --port $PORT`
   - Add env: `API_URL=https://supernal-backend.onrender.com`

4. **Frontend Service:**
   - Name: `supernal-frontend`
   - Build: `pip install -r requirements.txt`
   - Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Add env: `API_URL=https://supernal-backend.onrender.com` (use your backend URL)

5. **Done!** Your app will be live in ~5 minutes

---

## Alternative: Railway (Even Easier)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Railway auto-detects Python and deploys automatically!

---

## Environment Variables Needed

- `API_URL`: Your backend URL (for frontend)
- `PORT`: Automatically set by platform

---

## Important Notes

- ⚠️ First deployment may take 5-10 minutes (model downloads)
- 💾 Database and vector store persist automatically
- 🔄 Auto-purge runs on backend startup (7 days)
- 📦 All dependencies install automatically

---

## Need Help?

See `DEPLOYMENT.md` for detailed instructions for all platforms!

