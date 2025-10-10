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
import base64
from openai import OpenAI

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # service key for uploads
BUCKET = "couple-images"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)

def get_image_embedding(file_bytes: bytes):
    """Generate a vector embedding for an image using OpenAI CLIP."""
    img_b64 = base64.b64encode(file_bytes).decode("utf-8")
    response = client.embeddings.create(
        model="image-embedding-clip",
        input=img_b64
    )
    return response.data[0].embedding


@app.post("/upload-couple-images")
async def upload_couple_images(
    couple_id: str = Form(...),
    category: str = Form(...),
    files: list[UploadFile] = File(...)
):
    results = []

    for file in files:
        # 1) Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            logging.warning(f"Rejected file with invalid type: {file.filename}")
            results.append({
                "file_name": file.filename,
                "status": "error",
                "detail": "File type not allowed"
            })
            continue

        # 2) Read file bytes
        contents = await file.read()

        # 3) Check for duplicates
        existing = supabase.table("images").select("*")\
            .eq("couple_id", couple_id)\
            .eq("file_name", file.filename)\
            .eq("category", category)\
            .execute()

        duplicate_prevented = False
        new_row_created = False

        if existing.data:
            duplicate_prevented = True
            file_url = existing.data[0]["file_path"]
            logging.info(f"Duplicate prevented: {file.filename} for couple {couple_id}, category {category}")
        else:
            # 4) Upload to Supabase Storage
            path = f"{couple_id}/{category}/{uuid.uuid4()}_{file.filename}"
            try:
                res = supabase.storage.from_(BUCKET).upload(
                    path,
                    contents,
                    {"content-type": file.content_type}
                )
            except Exception as e:
                logging.error(f"Upload failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Supabase storage upload failed: {str(e)}"
                })
                continue

            file_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

            # 5) Generate embedding
            try:
                embedding = get_image_embedding(contents)
            except Exception as e:
                logging.error(f"Embedding generation failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Embedding generation failed: {str(e)}"
                })
                continue

            # 6) Insert metadata + embedding into images table
            try:
                supabase.table("images").insert({
                    "id": str(uuid.uuid4()),
                    "couple_id": couple_id,
                    "file_name": file.filename,
                    "file_path": file_url,
                    "category": category,
                    "embedding": embedding  # store as JSON or array depending on your column type
                }).execute()
                new_row_created = True
            except Exception as e:
                logging.error(f"Inserting metadata failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Inserting metadata failed: {str(e)}"
                })
                continue

        # 7) Append result for this file
        results.append({
            "file_name": file.filename,
            "file_url": file_url,
            "duplicate_prevented": duplicate_prevented,
            "new_row_created": new_row_created,
            "status": "success"
        })

    return {"results": results}



