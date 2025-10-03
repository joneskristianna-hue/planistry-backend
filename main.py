from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Create a Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Planistry Backend")

# -------------------------------
# Pydantic models
# -------------------------------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# -------------------------------
# Signup endpoint
# -------------------------------

@app.post("/signup")
async def signup(payload: SignupRequest):
    # Create user in Supabase Auth
    res = supabase.auth.sign_up({
        "email": payload.email,
        "password": payload.password
    })

    user = res.user
    if not user:
        raise HTTPException(status_code=400, detail="Signup failed")

    # Insert a row in the couples table
    supabase.table("couples").insert({
        "id": user.id,
        "name": payload.name,
        "email": payload.email,
        "wedding_date": None,
        "budget": 0,
        "guest_range": 0,
        "location": ""
    }).execute()

    # Return a clean response
    return {
        "message": "Signup successful",
        "user": {
            "id": user.id,
            "email": user.email,
            "confirmed_at": user.confirmed_at  # optional
        }
    }

# -------------------------------
# Login endpoint
# -------------------------------

@app.post("/login")
async def login(payload: LoginRequest):
    res = supabase.auth.sign_in_with_password({
        "email": payload.email,
        "password": payload.password
    })

    user = res.user
    if not user:
        raise HTTPException(status_code=400, detail="Login failed")

    return {
        "message": "Login successful",
        "user": {
            "id": user.id,
            "email": user.email,
            "confirmed_at": user.confirmed_at  # optional
        }
    }
