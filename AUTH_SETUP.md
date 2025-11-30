# Authentication Setup Guide

This document explains how to set up Gmail and phone-based authentication for the AI Document Q&A system.

## Features

✅ **Gmail/Google OAuth2 Authentication**
- Sign in with Google account
- Secure token-based authentication
- Automatic user creation

✅ **Phone Number Authentication**
- OTP (One-Time Password) verification
- SMS-based verification (currently logs to console)
- Secure phone number storage

✅ **JWT Token Management**
- Secure session tokens
- Token expiration (24 hours)
- Token verification

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

New dependencies added:
- `pyjwt>=2.8.0` - JWT token handling
- `cryptography>=41.0.0` - Cryptographic functions

### 2. Database Migration

The User model has been updated with new fields:
- `email` - User email address
- `phone` - User phone number
- `auth_type` - Authentication type ('email', 'phone', 'google', 'anonymous')
- `google_id` - Google OAuth ID
- `is_verified` - Verification status

**To update existing database:**

```python
# Run this once to update the database schema
from database import init_db
init_db()  # This will create new columns if they don't exist
```

Or delete `documents.db` to start fresh (⚠️ This will delete all data).

### 3. Environment Variables

Create or update `.env` file:

```env
# JWT Secret (auto-generated if not set, but recommended to set your own)
JWT_SECRET=your-secret-key-here

# Google OAuth (Optional - for Gmail login)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8501/auth/callback

# For production, use your actual domain
# GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/callback
```

### 4. Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable "Google+ API" or "Google Identity API"
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
5. Configure OAuth consent screen
6. Add authorized redirect URI: `http://localhost:8501/auth/callback`
7. Copy Client ID and Client Secret to `.env`

### 5. Phone Verification Setup

Currently, OTP codes are logged to console for testing. For production:

**Option 1: Use Twilio (Recommended)**
```python
# In auth.py, update send_otp_sms function:
from twilio.rest import Client

def send_otp_sms(phone: str, code: str) -> bool:
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"Your verification code is: {code}",
        from_=TWILIO_PHONE_NUMBER,
        to=phone
    )
    return message.sid is not None
```

**Option 2: Use AWS SNS**
```python
import boto3

def send_otp_sms(phone: str, code: str) -> bool:
    sns = boto3.client('sns')
    sns.publish(
        PhoneNumber=phone,
        Message=f"Your verification code is: {code}"
    )
    return True
```

## Usage

### For Users

1. **Gmail Login:**
   - Click "Sign in with Google"
   - Enter Google OAuth token (for testing)
   - Or integrate Google Sign-In button (requires frontend JS)

2. **Phone Login:**
   - Enter phone number
   - Click "Send Verification Code"
   - Check console/logs for OTP (in development)
   - Enter 6-digit code
   - Click "Verify Code"

3. **Guest Mode:**
   - Click "Continue as Guest"
   - Uses anonymous user ID

### API Endpoints

**POST `/auth/google`**
```json
{
  "token": "google_oauth_token"
}
```

**POST `/auth/phone/send-otp`**
```json
{
  "phone": "+1234567890"
}
```

**POST `/auth/phone/verify`**
```json
{
  "phone": "+1234567890",
  "code": "123456"
}
```

**POST `/auth/verify-token`**
```
token: jwt_token_string
```

**GET `/auth/user/{user_id}`**
Returns authenticated user information.

## Security Notes

1. **JWT Secret:** Use a strong, random secret in production
2. **HTTPS:** Always use HTTPS in production
3. **Token Storage:** Tokens are stored in Streamlit session state (client-side)
4. **Phone Numbers:** Normalized and stored securely
5. **OTP Expiration:** 10 minutes default
6. **OTP Attempts:** Maximum 3 attempts per OTP

## Testing

1. Start backend: `python3 -m uvicorn backend:app --reload`
2. Start frontend: `streamlit run app.py`
3. Test phone login (check console for OTP)
4. Test Google login (requires valid token)

## Production Deployment

1. Set strong `JWT_SECRET` in environment
2. Configure Google OAuth with production redirect URI
3. Integrate SMS service (Twilio/AWS SNS)
4. Enable HTTPS
5. Set up proper CORS policies
6. Use secure session storage

## Troubleshooting

**"Invalid Google token"**
- Check Google Client ID/Secret in `.env`
- Verify redirect URI matches Google Console settings
- Ensure token is not expired

**"OTP not received"**
- Check console logs (development mode)
- Verify phone number format
- Check SMS service configuration (production)

**"Token expired"**
- Tokens expire after 24 hours
- User needs to sign in again

## Next Steps

- [ ] Integrate Google Sign-In button in frontend
- [ ] Add SMS service integration (Twilio/AWS SNS)
- [ ] Add password reset functionality
- [ ] Add email verification
- [ ] Add 2FA support
- [ ] Add social login (Facebook, Twitter, etc.)

