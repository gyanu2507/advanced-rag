# Google OAuth Setup Guide

Complete step-by-step guide to configure Google Sign-In for the AI Document Q&A application.

## Prerequisites

- Google account
- Access to Google Cloud Console

## Step-by-Step Setup

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click **"New Project"**
4. Enter project name: `AI Document Q&A` (or any name)
5. Click **"Create"**

### 2. Configure OAuth Consent Screen

1. In the left sidebar, go to **"APIs & Services"** → **"OAuth consent screen"**
2. Select **"External"** user type (unless you have Google Workspace)
3. Click **"Create"**
4. Fill in the required information:
   - **App name**: `AI Document Q&A`
   - **User support email**: Your email address
   - **Developer contact information**: Your email address
5. Click **"Save and Continue"**
6. **Scopes** (Step 2):
   - Click **"Add or Remove Scopes"**
   - Select: `email`, `profile`, `openid`
   - Click **"Update"** → **"Save and Continue"**
7. **Test users** (Step 3):
   - Add your email if needed (for testing)
   - Click **"Save and Continue"**
8. **Summary** (Step 4):
   - Review and click **"Back to Dashboard"**

### 3. Create OAuth 2.0 Credentials

1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"+ CREATE CREDENTIALS"** → **"OAuth client ID"**
3. Select **"Web application"** as application type
4. Enter a name: `AI Document Q&A Web Client`
5. **Authorized JavaScript origins:**
   - `http://localhost:8501` (for local development)
   - `https://yourdomain.com` (for production)
6. **Authorized redirect URIs:**
   - `http://localhost:8501/auth/callback` (for local)
   - `https://yourdomain.com/auth/callback` (for production)
7. Click **"CREATE"**
8. **IMPORTANT:** Copy the **Client ID** and **Client Secret** immediately
   - You won't be able to see the secret again!

### 4. Update Environment Variables

Edit your `.env` file in the project root:

```env
# Google OAuth Credentials
GOOGLE_CLIENT_ID=your-client-id-here.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret-here
GOOGLE_REDIRECT_URI=http://localhost:8501/auth/callback

# JWT Secret
JWT_SECRET=your-jwt-secret-here
```

### 5. Verify Configuration

Run this command to verify:

```bash
python3 -c "from auth import GOOGLE_OAUTH_ENABLED, GOOGLE_CLIENT_ID; print(f'Enabled: {GOOGLE_OAUTH_ENABLED}'); print(f'Client ID: {GOOGLE_CLIENT_ID[:30]}...')"
```

Expected output:
```
Enabled: True
Client ID: 335692574178-u63ln8lkr9r5603m4mbeotbvtu2kv7dk...
```

### 6. Restart Application

```bash
# Stop current servers (Ctrl+C)
# Then restart:
python3 -m uvicorn backend:app --reload
streamlit run app.py
```

### 7. Test Google Sign-In

1. Open the application in your browser
2. Go to the login page
3. Click on **"📧 Email (Google)"** tab
4. You should see the Google Sign-In button
5. Click it and sign in with your Google account
6. You should be automatically redirected and signed in!

## Troubleshooting

### "Google OAuth Not Configured" Warning

- Check that `.env` file exists and has correct values
- Verify environment variables are loaded: `python3 -c "from auth import GOOGLE_CLIENT_ID; print(GOOGLE_CLIENT_ID)"`
- Restart the application after updating `.env`

### "redirect_uri_mismatch" Error

- Verify redirect URI in Google Cloud Console matches exactly:
  - `http://localhost:8501/auth/callback` (no trailing slash)
- Check authorized JavaScript origins include:
  - `http://localhost:8501` (no trailing slash)

### "Invalid Client" Error

- Verify Client ID and Client Secret are correct
- Check that OAuth consent screen is published (or add test users)
- Ensure the OAuth client is enabled in Google Cloud Console

### Button Not Showing

- Check browser console for JavaScript errors
- Verify Google Sign-In JavaScript SDK is loading
- Check that Client ID is being passed correctly

### Authentication Fails

- Check backend logs for errors
- Verify token is being received
- Check that `GOOGLE_CLIENT_SECRET` is set correctly

## Production Deployment

For production, update:

1. **Google Cloud Console:**
   - Add production redirect URI: `https://yourdomain.com/auth/callback`
   - Add production JavaScript origin: `https://yourdomain.com`

2. **Environment Variables:**
   ```env
   GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/callback
   ```

3. **HTTPS Required:**
   - Google OAuth requires HTTPS in production
   - Use a service like Let's Encrypt for SSL certificates

## Security Notes

- ✅ Never commit `.env` file to Git (already in `.gitignore`)
- ✅ Keep Client Secret secure
- ✅ Use different credentials for development and production
- ✅ Regularly rotate secrets
- ✅ Monitor OAuth usage in Google Cloud Console

## Current Configuration Status

After setup, verify your configuration:
- ✅ Client ID: Should start with numbers and end with `.apps.googleusercontent.com`
- ✅ Client Secret: Should start with `GOCSPX-`
- ✅ JWT Secret: Any secure random string
- ✅ Redirect URI: `http://localhost:8501/auth/callback` (for local)

**Status:** Once configured, you're ready to use! 🎉

