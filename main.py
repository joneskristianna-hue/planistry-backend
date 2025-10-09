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
    email: str #temporarily no email validation for local testing, UPDATE BEFORE PROD
    password: str
    name: str | None = None

class LoginRequest(BaseModel):
    email: str #temporarily no email validation for local testing, UPDATE BEFORE PROD
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

    if res.user is None:
        raise HTTPException(status_code=400, detail="Signup failed")

    user = res.user

    # Insert into couples table with enhanced error handling
    from postgrest.exceptions import APIError

    try:
        supabase.table("couples").insert({
            "id": user.id,
            "name": payload.name,
            "email": payload.email,
            "wedding_date": None,
            "budget": 0,
            "guest_range": 0,
            "location": ""
        }).execute()
    except APIError as e:
        if 'duplicate key value violates unique constraint "couples_email_key"' in str(e):
            raise HTTPException(
                status_code=400,
                detail="This email is already registered. Try logging in instead."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected database error: {str(e)}"
            )

    return {
        "message": "Signup successful",
        "user": {
            "id": user.id,
            "email": user.email,
            "confirmed_at": user.confirmed_at
        }
    }

# -------------------------------
# Login endpoint
# -------------------------------

from supabase import create_client, Client

from fastapi import HTTPException

@app.post("/login")
async def login(payload: LoginRequest):
    email = payload.email
    password = payload.password
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        user = res.user
        if user is None:
            # This is just in case Supabase returns None without raising
            raise HTTPException(status_code=401, detail="Email or password is incorrect")
        
        return {
            "message": "Login successful",
            "user": {
                "id": user.id,
                "email": user.email,
                "confirmed_at": user.confirmed_at
            }
        }

    except Exception as e:
        # Catch the AuthApiError and map it to your friendly message
        if "Invalid login credentials" in str(e):
            raise HTTPException(status_code=401, detail="Email or password is incorrect")
        else:
            raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")

# -------------------------------
# happy path home page
# -------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Planistry Backend!"}

# -------------------------------
# Upload Couple Image endpoint
# -------------------------------
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from supabase import create_client
import uuid
import os
import logging

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET = "couple-images"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure basic logging
logging.basicConfig(level=logging.INFO)

@app.post("/upload-couple-image")
async def upload_couple_image(couple_id: str = Form(...), file: UploadFile = File(...)):
    # 1) Validate file type
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="File type not allowed")

    # 2) Read file bytes
    contents = await file.read()

    # 3) Check if this couple has already uploaded a file with the same name
    existing = supabase.table("images").select("*")\
        .eq("couple_id", couple_id).eq("file_name", file.filename).execute()

    duplicate_prevented = False
    new_row_created = False
    if existing.data:
        duplicate_prevented = True
        logging.info(f"Duplicate upload prevented for couple {couple_id}, file '{file.filename}'")

    # 4) Only proceed with upload if it’s not a duplicate
    if not duplicate_prevented:
        path = f"{couple_id}/{uuid.uuid4()}_{file.filename}"

        # 5) Upload to Supabase Storage
        try:
            res = supabase.storage.from_(BUCKET).upload(
                path,
                contents,
                {"content-type": file.content_type}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Supabase storage upload failed: {str(e)}")

        # 6) Construct public URL (if bucket is public)
        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

        # 7) Insert metadata into images table
        try:
            supabase.table("images").insert({
                "id": str(uuid.uuid4()),
                "couple_id": couple_id,  # must be a valid UUID
                "file_name": file.filename,
                "file_path": file_url
            }).execute()
            new_row_created = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to insert into images table: {str(e)}")
    else:
        file_url = existing.data[0]["file_path"]  # return the URL of the existing file

    # 8) Return success response, including info if a duplicate was skipped
    response = {
        "status": "success",
        "file_url": file_url,
        "duplicate_prevented": duplicate_prevented
        "new_row_created": new_row_created
    }

    if duplicate_prevented:
        logging.info(f"Upload request for couple {couple_id} skipped duplicate file '{file.filename}'")

    return response

