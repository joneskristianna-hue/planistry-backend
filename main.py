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
    vendor_type: str = Form("florist"),  # NEW: florist, photographer, venue, etc.
    category: str = Form("florals"),
    city: str = Form("Austin"),  # NEW: for multi-city support
    state: str = Form("TX"),  # NEW: for state-level filtering
    files: list[UploadFile] = File(...)
):
    """
    Upload vendor portfolio images and generate embeddings.
    Generic endpoint that works for any vendor type.
    """
    results = []

    for file in files:
        # 1) Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            logging.warning(f"Rejected file with invalid type: {file.filename}")
            results.append({
                "file_name": file.filename,
                "status": "error",
                "detail": "File type not allowed. Use PNG, JPG, JPEG, or WEBP"
            })
            continue

        # 2) Read file bytes
        contents = await file.read()

        # 3) Upload to Supabase Storage
        # Organize by vendor type: vendors/{vendor_type}/{vendor_id}/{category}/
        path = f"vendors/{vendor_type}/{vendor_id}/{category}/{uuid.uuid4()}_{file.filename}"
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
                "detail": f"Storage upload failed: {str(e)}"
            })
            continue

        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

        # 4) Generate embedding
        try:
            embedding = get_image_embedding(contents)
            logging.info(f"Generated embedding for {vendor_type} image: {file.filename}")
        except Exception as e:
            logging.error(f"Embedding generation failed for {file.filename}: {str(e)}")
            results.append({
                "file_name": file.filename,
                "status": "error",
                "detail": f"Embedding generation failed: {str(e)}"
            })
            continue

        # 5) Generate unique ID
        image_id = str(uuid.uuid4())

        # 6) Insert into Pinecone with COMPLETE metadata
        try:
            index.upsert(vectors=[(
                image_id,
                embedding,
                {
                    # Vendor identification
                    "vendor_id": vendor_id,
                    "vendor_type": vendor_type,  # ← CRITICAL for filtering!
                    
                    # Categorization
                    "category": category,
                    
                    # Location (for multi-city expansion)
                    "city": city,
                    "state": state,
                    
                    # File info
                    "file_path": file_url,
                    "file_name": file.filename,
                    
                    # Type marker
                    "type": "vendor_image"  # vs "couple_image"
                }
            )])
            logging.info(f"Inserted {vendor_type} image into Pinecone: {file.filename} (city: {city})")
        except Exception as e:
            logging.error(f"Pinecone insert failed for {file.filename}: {str(e)}")
            results.append({
                "file_name": file.filename,
                "status": "error",
                "detail": f"Pinecone insert failed: {str(e)}"
            })
            continue

        # 7) Insert metadata into Supabase
        try:
            supabase.table("vendor_images").insert({
                "id": image_id,
                "vendor_id": vendor_id,
                "file_name": file.filename,
                "file_path": file_url,
                "category": category,
                "embedding_id": image_id
            }).execute()
            logging.info(f"Inserted {vendor_type} image metadata into Supabase")
        except Exception as e:
            logging.error(f"Database insert failed for {file.filename}: {str(e)}")
            results.append({
                "file_name": file.filename,
                "status": "error",
                "detail": f"Database insert failed: {str(e)}"
            })
            continue

        # Success!
        results.append({
            "file_name": file.filename,
            "file_url": file_url,
            "status": "success"
        })

    return {
        "results": results,
        "vendor_id": vendor_id,
        "vendor_type": vendor_type,
        "category": category,
        "location": f"{city}, {state}",
        "total_uploaded": len([r for r in results if r["status"] == "success"]),
        "total_failed": len([r for r in results if r["status"] == "error"])
    }

# -------------------------------
# Vendor matching endpoint
# ------------------------------- 
@app.post("/find-matching-vendors")
async def find_matching_vendors(
    couple_id: str = Form(...),
    vendor_type: str = Form("florist"),  # florist, photographer, venue, caterer, planner, dj
    category: str = Form(None),  # Optional: bouquet, centerpiece, etc.
    city: str = Form("Austin"),  # For future expansion
    state: str = Form("TX"),
    top_k: int = Form(10),
    min_match_score: float = Form(0.7)  # Filter out low matches
):
    """
    Find vendors whose work matches the couple's aesthetic.
    Generic endpoint that works for any vendor type.
    Returns rich vendor profiles with match percentages (The Knot style).
    """
    try:
        # 1. Get couple's image embeddings from Supabase
        query = supabase.table("images").select("id, embedding_id, category").eq("couple_id", couple_id)
        
        if category:
            query = query.eq("category", category)
        
        couple_images = query.execute()
        
        if not couple_images.data:
            raise HTTPException(
                status_code=404, 
                detail=f"No images found for couple {couple_id}"
            )
        
        logging.info(f"Found {len(couple_images.data)} images for couple {couple_id}")
        
        # 2. Fetch embeddings from Pinecone
        embedding_ids = [img["embedding_id"] for img in couple_images.data]
        fetch_response = index.fetch(ids=embedding_ids)
        couple_embeddings = [
            fetch_response.vectors[id].values 
            for id in embedding_ids 
            if id in fetch_response.vectors
        ]
        
        if not couple_embeddings:
            raise HTTPException(status_code=404, detail="Could not fetch embeddings from Pinecone")
        
        # 3. Calculate average embedding (their "ideal aesthetic")
        avg_embedding = np.mean(couple_embeddings, axis=0).tolist()
        logging.info(f"Calculated average aesthetic embedding for couple {couple_id}")
        
        # 4. Query Pinecone for similar vendor images
        search_filter = {
            "type": "vendor_image",
            "vendor_type": vendor_type,  # Dynamic vendor type filtering
            "city": city
        }
        
        if category:
            search_filter["category"] = category
        
        results = index.query(
            vector=avg_embedding,
            top_k=top_k * 10,  # Get more results to group by vendor
            filter=search_filter,
            include_metadata=True
        )
        
        if not results.matches:
            return {
                "couple_id": couple_id,
                "vendor_type": vendor_type,
                "category": category,
                "location": f"{city}, {state}",
                "matching_vendors": [],
                "message": f"No matching {vendor_type}s found. We're onboarding more {city} {vendor_type}s!"
            }
        
        # 5. Group results by vendor and calculate match scores
        vendor_data = {}
        
        for match in results.matches:
            vendor_id = match.metadata.get("vendor_id")
            if not vendor_id or match.score < min_match_score:
                continue
            
            if vendor_id not in vendor_data:
                vendor_data[vendor_id] = {
                    "scores": [],
                    "sample_images": []
                }
            
            vendor_data[vendor_id]["scores"].append(match.score)
            
            # Keep top 4 sample images per vendor
            if len(vendor_data[vendor_id]["sample_images"]) < 4:
                vendor_data[vendor_id]["sample_images"].append({
                    "file_path": match.metadata.get("file_path"),
                    "category": match.metadata.get("category"),
                    "similarity_score": round(match.score, 3)
                })
        
        # 6. Get full vendor profiles from database
        vendor_ids = list(vendor_data.keys())
        
        vendors_response = supabase.table("vendors").select(
            "id, business_name, tagline, description, instagram_handle, website, "
            "price_range, city, state, verified, years_in_business, vendor_type"
        ).in_("id", vendor_ids)\
         .eq("is_active", True)\
         .eq("vendor_type", vendor_type)\
         .eq("city", city)\
         .execute()
        
        # 7. Get review statistics for each vendor
        reviews_response = supabase.table("vendor_reviews").select(
            "vendor_id, rating"
        ).in_("vendor_id", vendor_ids).eq("is_visible", True).execute()
        
        # Calculate average ratings per vendor
        vendor_ratings = {}
        for review in reviews_response.data:
            vid = review["vendor_id"]
            if vid not in vendor_ratings:
                vendor_ratings[vid] = []
            vendor_ratings[vid].append(review["rating"])
        
        # 8. Combine match scores with vendor profiles
        matching_vendors = []
        
        for vendor in vendors_response.data:
            vendor_id = vendor["id"]
            scores = vendor_data[vendor_id]["scores"]
            avg_score = np.mean(scores)
            match_percentage = round(avg_score * 100, 1)  # Convert to percentage
            
            # Calculate average rating
            ratings = vendor_ratings.get(vendor_id, [])
            avg_rating = round(np.mean(ratings), 1) if ratings else None
            review_count = len(ratings)
            
            matching_vendors.append({
                # Match Info
                "match_percentage": match_percentage,
                "match_rank": None,  # Will be set after sorting
                "num_matching_images": len(scores),
                
                # Vendor Profile
                "vendor_id": vendor_id,
                "business_name": vendor["business_name"],
                "vendor_type": vendor["vendor_type"],  # Added for clarity
                "tagline": vendor.get("tagline"),
                "description": vendor["description"],
                "instagram_handle": vendor.get("instagram_handle"),
                "website": vendor.get("website"),
                "price_range": vendor.get("price_range"),
                "location": f"{vendor['city']}, {vendor['state']}",
                "verified": vendor.get("verified", False),
                "years_in_business": vendor.get("years_in_business"),
                
                # Social Proof
                "avg_rating": avg_rating,
                "review_count": review_count,
                
                # Sample Images
                "sample_images": vendor_data[vendor_id]["sample_images"]
            })
        
        # 9. Sort by match percentage and assign ranks
        matching_vendors.sort(key=lambda x: x["match_percentage"], reverse=True)
        
        for rank, vendor in enumerate(matching_vendors[:top_k], start=1):
            vendor["match_rank"] = rank
        
        # 10. Log match for analytics
        for vendor in matching_vendors[:top_k]:
            try:
                supabase.table("vendor_matches").insert({
                    "couple_id": couple_id,
                    "vendor_id": vendor["vendor_id"],
                    "match_score": vendor["match_percentage"] / 100,
                    "match_rank": vendor["match_rank"],
                    "matched_category": category,
                    "num_images_compared": len(couple_embeddings)
                }).execute()
            except Exception as e:
                logging.warning(f"Failed to log match: {str(e)}")
        
        logging.info(f"Found {len(matching_vendors)} matching {vendor_type}s for couple {couple_id}")
        
        return {
            "couple_id": couple_id,
            "vendor_type": vendor_type,
            "category": category,
            "location": f"{city}, {state}",
            "num_couple_images_analyzed": len(couple_embeddings),
            "matching_vendors": matching_vendors[:top_k],
            "total_vendors_found": len(matching_vendors)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Vendor matching failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")

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


