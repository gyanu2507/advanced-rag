# 🤗 Hugging Face Spaces Deployment

## Quick Start

1. **Go to [Hugging Face Spaces](https://huggingface.co/spaces)**
2. **Click "Create new Space"**
3. **Select:**
   - SDK: **Streamlit**
   - Name: `advanced-rag`
   - Visibility: Public
4. **Import from GitHub**: `gyanu2507/advanced-rag`
5. **Set Environment Variable:**
   - `API_URL`: Your backend URL (deploy backend on Railway/Fly.io first)
6. **Wait for build** (~10-15 minutes)
7. **Done!** Your app is live! 🎉

## Backend Deployment Required

Since Hugging Face Spaces only hosts frontend, deploy backend separately:

### Railway (Recommended):
```bash
npm i -g @railway/cli
railway login
railway init
railway up
```

### Fly.io:
```bash
fly auth login
fly launch
fly deploy
```

## Your Live URLs

- **Frontend**: `https://huggingface.co/spaces/YOUR_USERNAME/advanced-rag`
- **Backend**: `https://your-backend.railway.app`

See `HUGGINGFACE_DEPLOY.md` for detailed instructions!

