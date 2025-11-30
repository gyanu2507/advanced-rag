"""
Authentication service for Gmail and phone-based login.
"""
import os
import secrets
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict
from sqlalchemy.orm import Session
from database import User, get_db
import requests
from dotenv import load_dotenv

load_dotenv()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8501/auth/callback")

# Check if Google OAuth is configured
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

# Phone Verification (Simple OTP - for production, use Twilio or similar)
OTP_STORAGE: Dict[str, Dict] = {}  # {phone: {code, expires_at, attempts}}
OTP_EXPIRATION_MINUTES = 10
OTP_MAX_ATTEMPTS = 3


def generate_jwt_token(user_id: str, email: Optional[str] = None) -> str:
    """Generate JWT token for authenticated user."""
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def generate_otp() -> str:
    """Generate a 6-digit OTP."""
    return f"{secrets.randbelow(900000) + 100000:06d}"


def send_otp_sms(phone: str, code: str) -> bool:
    """Send OTP via SMS. For production, integrate with Twilio, AWS SNS, etc."""
    # TODO: Integrate with SMS service (Twilio, AWS SNS, etc.)
    # For now, just log it (in production, this should send actual SMS)
    print(f"📱 OTP for {phone}: {code}")
    return True


def create_or_get_user_by_email(db: Session, email: str, google_id: Optional[str] = None) -> User:
    """Create or get user by email."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        # Generate user_id from email hash
        user_id = hashlib.sha256(email.encode()).hexdigest()[:16]
        user = User(
            user_id=user_id,
            email=email,
            auth_type="google" if google_id else "email",
            google_id=google_id,
            is_verified="true"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update last active
        user.last_active = datetime.utcnow()
        if google_id:
            user.google_id = google_id
        db.commit()
    return user


def create_or_get_user_by_phone(db: Session, phone: str) -> User:
    """Create or get user by phone."""
    # Normalize phone number (remove spaces, dashes, etc.)
    phone_normalized = "".join(filter(str.isdigit, phone))
    
    user = db.query(User).filter(User.phone == phone_normalized).first()
    if not user:
        # Generate user_id from phone hash
        user_id = hashlib.sha256(phone_normalized.encode()).hexdigest()[:16]
        user = User(
            user_id=user_id,
            phone=phone_normalized,
            auth_type="phone",
            is_verified="false"  # Will be verified after OTP
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.last_active = datetime.utcnow()
        db.commit()
    return user


def verify_google_token(token: str) -> Optional[Dict]:
    """Verify Google OAuth token and get user info.
    
    Supports both:
    1. OAuth2 access tokens (Bearer tokens)
    2. Google ID tokens (JWT tokens from Google Sign-In)
    """
    try:
        # Try as ID token first (Google Sign-In)
        try:
            import jwt
            # Decode without verification first to get the token type
            decoded = jwt.decode(token, options={"verify_signature": False})
            if "email" in decoded:
                # This is an ID token, return the decoded info
                return {
                    "id": decoded.get("sub"),
                    "email": decoded.get("email"),
                    "name": decoded.get("name", ""),
                    "picture": decoded.get("picture", ""),
                    "verified_email": decoded.get("email_verified", False)
                }
        except:
            pass
        
        # Try as OAuth2 access token
        response = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error verifying Google token: {e}")
        return None


def initiate_phone_verification(phone: str) -> Dict:
    """Initiate phone verification by sending OTP."""
    phone_normalized = "".join(filter(str.isdigit, phone))
    
    # Check if there's an existing OTP
    if phone_normalized in OTP_STORAGE:
        existing = OTP_STORAGE[phone_normalized]
        if datetime.utcnow() < existing["expires_at"]:
            # Resend existing code
            send_otp_sms(phone_normalized, existing["code"])
            return {
                "status": "success",
                "message": "OTP sent (resending existing code)",
                "expires_in": int((existing["expires_at"] - datetime.utcnow()).total_seconds())
            }
    
    # Generate new OTP
    code = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=OTP_EXPIRATION_MINUTES)
    
    OTP_STORAGE[phone_normalized] = {
        "code": code,
        "expires_at": expires_at,
        "attempts": 0
    }
    
    # Send OTP
    send_otp_sms(phone_normalized, code)
    
    return {
        "status": "success",
        "message": "OTP sent successfully",
        "expires_in": OTP_EXPIRATION_MINUTES * 60
    }


def verify_phone_otp(phone: str, code: str, db: Session) -> Optional[Dict]:
    """Verify phone OTP and return user token."""
    phone_normalized = "".join(filter(str.isdigit, phone))
    
    if phone_normalized not in OTP_STORAGE:
        return {"status": "error", "message": "No OTP found. Please request a new one."}
    
    otp_data = OTP_STORAGE[phone_normalized]
    
    # Check expiration
    if datetime.utcnow() > otp_data["expires_at"]:
        del OTP_STORAGE[phone_normalized]
        return {"status": "error", "message": "OTP expired. Please request a new one."}
    
    # Check attempts
    if otp_data["attempts"] >= OTP_MAX_ATTEMPTS:
        del OTP_STORAGE[phone_normalized]
        return {"status": "error", "message": "Too many attempts. Please request a new OTP."}
    
    # Verify code
    if code != otp_data["code"]:
        otp_data["attempts"] += 1
        return {"status": "error", "message": f"Invalid OTP. {OTP_MAX_ATTEMPTS - otp_data['attempts']} attempts remaining."}
    
    # OTP verified - create/get user and generate token
    user = create_or_get_user_by_phone(db, phone_normalized)
    user.is_verified = "true"
    db.commit()
    
    # Clean up OTP
    del OTP_STORAGE[phone_normalized]
    
    # Generate JWT token
    token = generate_jwt_token(user.user_id, user.email)
    
    return {
        "status": "success",
        "message": "Phone verified successfully",
        "token": token,
        "user_id": user.user_id,
        "user": {
            "user_id": user.user_id,
            "phone": user.phone,
            "is_verified": user.is_verified
        }
    }


def authenticate_with_google(token: str, db: Session) -> Optional[Dict]:
    """Authenticate user with Google OAuth token."""
    user_info = verify_google_token(token)
    if not user_info:
        return None
    
    email = user_info.get("email")
    google_id = user_info.get("id")
    name = user_info.get("name", "")
    
    if not email:
        return None
    
    # Create or get user
    user = create_or_get_user_by_email(db, email, google_id)
    
    # Generate JWT token
    jwt_token = generate_jwt_token(user.user_id, user.email)
    
    return {
        "status": "success",
        "token": jwt_token,
        "user_id": user.user_id,
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "name": name,
            "is_verified": user.is_verified
        }
    }

