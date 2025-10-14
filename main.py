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
# Upload Couple Image endpoint + Image Embedding Generation
# -------------------------------
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from supabase import create_client
import uuid
import os
import logging
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np
from pinecone import Pinecone

app = FastAPI()

# Load CLIP model once when server starts
model = SentenceTransformer('clip-ViT-B-32')

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("planistry-image-embeddings")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service key for uploads
BUCKET = "couple-images"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)

def get_image_embedding(file_bytes: bytes):
    """Generate 512-dimensional CLIP embedding"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        embedding = model.encode(image, convert_to_numpy=True)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    except Exception as e:
        raise Exception(f"Embedding generation failed: {str(e)}")

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
                logging.info(f"Generated embedding for {file.filename}: {len(embedding)} dimensions")
            except Exception as e:
                logging.error(f"Embedding generation failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Embedding generation failed: {str(e)}"
                })
                continue

            # 6) Generate unique ID for this image
            image_id = str(uuid.uuid4())

            # 7) Insert into Pinecone
            try:
                index.upsert(vectors=[(
                    image_id,  # unique ID
                    embedding,  # 512-dim vector
                    {
                        "couple_id": couple_id,
                        "category": category,
                        "file_path": file_url,
                        "file_name": file.filename,
                        "type": "couple_image"  # vs "vendor_image"
                    }
                )])
                logging.info(f"Inserted {file.filename} into Pinecone with ID {image_id}")
            except Exception as e:
                logging.error(f"Pinecone insert failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Pinecone insert failed: {str(e)}"
                })
                continue

            # 8) Insert metadata into Supabase images table
            try:
                supabase.table("images").insert({
                    "id": image_id,  # Use same ID as Pinecone
                    "couple_id": couple_id,
                    "file_name": file.filename,
                    "file_path": file_url,
                    "category": category,
                    "embedding_id": image_id  # Reference to Pinecone vector ID
                }).execute()
                new_row_created = True
                logging.info(f"Inserted {file.filename} metadata into Supabase")
            except Exception as e:
                logging.error(f"Inserting metadata failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Inserting metadata failed: {str(e)}"
                })
                continue

        # 9) Append result for this file
        results.append({
            "file_name": file.filename,
            "file_url": file_url,
            "duplicate_prevented": duplicate_prevented,
            "new_row_created": new_row_created,
            "status": "success"
        })

    return {
        "results": results,
        "total_uploaded": len([r for r in results if r["status"] == "success"]),
        "total_failed": len([r for r in results if r["status"] == "error"])
    }
    
# -------------------------------
# Upload Vendor Images endpoint + Image Embedding Generation
# -------------------------------    
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from supabase import create_client
import uuid
import os
import logging
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import numpy as np
from pinecone import Pinecone

app = FastAPI()

# Load CLIP model once when server starts
model = SentenceTransformer('clip-ViT-B-32')

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("planistry-image-embeddings")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service key for uploads
BUCKET = "vendor-images"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)

def get_image_embedding(file_bytes: bytes):
    """Generate 512-dimensional CLIP embedding"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        embedding = model.encode(image, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    except Exception as e:
        raise Exception(f"Embedding generation failed: {str(e)}")

@app.post("/upload-vendor-images")
async def upload_vendor_images(
    vendor_id: str = Form(...),
    category: str = Form(...),
    files: list[UploadFile] = File(...)
):
    results = []

    for file in files:
        # Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            logging.warning(f"Rejected file with invalid type: {file.filename}")
            results.append({
                "file_name": file.filename,
                "status": "error",
                "detail": "File type not allowed"
            })
            continue

        contents = await file.read()

        # Check for duplicates
        existing = supabase.table("vendor_images").select("*")\
            .eq("vendor_id", vendor_id)\
            .eq("file_name", file.filename)\
            .eq("category", category)\
            .execute()

        duplicate_prevented = False
        new_row_created = False

        if existing.data:
            duplicate_prevented = True
            file_url = existing.data[0]["file_path"]
            logging.info(f"Duplicate prevented: {file.filename} for vendor {vendor_id}, category {category}")
        else:
            # Upload to Supabase Storage
            path = f"{vendor_id}/{category}/{uuid.uuid4()}_{file.filename}"
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

            # Generate embedding
            try:
                embedding = get_image_embedding(contents)
                logging.info(f"Generated embedding for {file.filename}: {len(embedding)} dimensions")
            except Exception as e:
                logging.error(f"Embedding generation failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Embedding generation failed: {str(e)}"
                })
                continue

            image_id = str(uuid.uuid4())

            # Insert into Pinecone
            try:
                index.upsert(vectors=[(
                    image_id,
                    embedding,
                    {
                        "vendor_id": vendor_id,
                        "category": category,
                        "file_path": file_url,
                        "file_name": file.filename,
                        "type": "vendor_image"
                    }
                )])
                logging.info(f"Inserted {file.filename} into Pinecone with ID {image_id}")
            except Exception as e:
                logging.error(f"Pinecone insert failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Pinecone insert failed: {str(e)}"
                })
                continue

            # Insert metadata into Supabase vendor_images table
            try:
                supabase.table("vendor_images").insert({
                    "id": image_id,
                    "vendor_id": vendor_id,
                    "file_name": file.filename,
                    "file_path": file_url,
                    "category": category,
                    "embedding_id": image_id
                }).execute()
                new_row_created = True
                logging.info(f"Inserted {file.filename} metadata into vendor_images")
            except Exception as e:
                logging.error(f"Inserting metadata failed for {file.filename}: {str(e)}")
                results.append({
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Inserting metadata failed: {str(e)}"
                })
                continue

        results.append({
            "file_name": file.filename,
            "file_url": file_url,
            "duplicate_prevented": duplicate_prevented,
            "new_row_created": new_row_created,
            "status": "success"
        })

    return {
        "results": results,
        "total_uploaded": len([r for r in results if r["status"] == "success"]),
        "total_failed": len([r for r in results if r["status"] == "error"])
    }


@app.get("/health")
async def health_check():
    # Test Pinecone connection
    try:
        index.describe_index_stats()
        pinecone_healthy = True
    except:
        pinecone_healthy = False
    
    # Test Supabase connection
    try:
        supabase.table("images").select("id").limit(1).execute()
        supabase_healthy = True
    except:
        supabase_healthy = False
    
    return {
        "status": "healthy" if (pinecone_healthy and supabase_healthy) else "degraded",
        "model": "clip-ViT-B-32",
        "embedding_dimension": 512,
        "pinecone_index": "planistry-image-embeddings",
        "pinecone_connected": pinecone_healthy,
        "supabase_connected": supabase_healthy
    }


