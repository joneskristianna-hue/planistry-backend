# ===================================
# IMPORTS (All at the top)
# ===================================
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
from supabase import create_client, Client
from postgrest.exceptions import APIError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PIL import Image
from pinecone import Pinecone
import os
import uuid
import logging
import io
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
from typing import Optional


# Load environment variables
load_dotenv()

# ===================================
# CONFIGURATION
# ===================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ===================================
# INITIALIZE APP & SERVICES (Once!)
# ===================================
app = FastAPI(title="Planistry Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "https://planistry.vercel.app",  # Your production frontend (we'll deploy here)
        "https://*.vercel.app",  # All Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set up model caching (works locally and on Render)
if os.path.exists("/opt/render"):
    # Running on Render
    CACHE_DIR = "/opt/render/project/.cache"
else:
    # Running locally
    CACHE_DIR = os.path.join(os.getcwd(), ".cache")

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

# Load CLIP model once when server starts (with caching)
model = SentenceTransformer("clip-ViT-B-32", cache_folder=CACHE_DIR)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("planistry-image-embeddings")

# Logging configuration
logging.basicConfig(level=logging.INFO)


# ===================================
# PYDANTIC MODELS
# ===================================
class SignupRequest(BaseModel):
    email: str
    password: str
    name: str | None = None
    wedding_date: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


# ===================================
# HELPER FUNCTIONS
# ===================================
def get_image_embedding(file_bytes: bytes):
    """Generate 512-dimensional CLIP embedding"""
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Let SentenceTransformer handle normalization internally
        embedding = model.encode(
            image,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Built-in normalization
            show_progress_bar=False,
        )

        return embedding.tolist()
    except Exception as e:
        raise Exception(f"Embedding generation failed: {str(e)}")


# ===================================
# AUTH ENDPOINTS
# ===================================
@app.post("/signup")
async def signup(payload: SignupRequest):
    """Create new couple account"""
    # Create user in Supabase Auth
    res = supabase.auth.sign_up({"email": payload.email, "password": payload.password})

    if res.user is None:
        raise HTTPException(status_code=400, detail="Signup failed")

    user = res.user
    session = res.session

    # Insert into couples table
    try:
        supabase.table("couples").insert(
            {
                "id": user.id,
                "name": payload.name,
                "email": payload.email,
                "wedding_date": payload.wedding_date,  # Now accepts the date from form
                "budget": 0,
                "guest_range": 0,
                "location": "",
            }
        ).execute()
    except APIError as e:
        if 'duplicate key value violates unique constraint "couples_email_key"' in str(
            e
        ):
            raise HTTPException(
                status_code=400,
                detail="This email is already registered. Try logging in instead.",
            )
        else:
            raise HTTPException(
                status_code=500, detail=f"Unexpected database error: {str(e)}"
            )

    return {
        "message": "Signup successful",
        "access_token": session.access_token,
        "user": {"id": user.id, "email": user.email, "confirmed_at": user.confirmed_at},
    }


@app.post("/login")
async def login(payload: LoginRequest):
    """Login existing couple"""
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": payload.email, "password": payload.password}
        )
        user = res.user
        session = res.session

        if user is None:
            raise HTTPException(
                status_code=401, detail="Email or password is incorrect"
            )

        return {
            "message": "Login successful",
            "access_token": session.access_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "confirmed_at": user.confirmed_at,
            },
        }
    except Exception as e:
        if "Invalid login credentials" in str(e):
            raise HTTPException(
                status_code=401, detail="Email or password is incorrect"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")


# ===================================
# IMAGE UPLOAD ENDPOINTS
# ===================================
@app.post("/upload-couple-images")
async def upload_couple_images(
    couple_id: str = Form(...),
    category: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload couple inspiration images and generate embeddings"""
    BUCKET = "couple-images"
    results = []

    for file in files:
        # 1) Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            logging.warning(f"Rejected file with invalid type: {file.filename}")
            results.append(
                {
                    "file_name": file.filename,
                    "status": "error",
                    "detail": "File type not allowed",
                }
            )
            continue

        # 2) Read file bytes
        contents = await file.read()

        # 3) Check for duplicates
        existing = (
            supabase.table("images")
            .select("*")
            .eq("couple_id", couple_id)
            .eq("file_name", file.filename)
            .eq("category", category)
            .execute()
        )

        duplicate_prevented = False
        new_row_created = False

        if existing.data:
            duplicate_prevented = True
            file_url = existing.data[0]["file_path"]
            logging.info(f"Duplicate prevented: {file.filename} for couple {couple_id}")
        else:
            # 4) Upload to Supabase Storage
            path = f"{couple_id}/{category}/{uuid.uuid4()}_{file.filename}"
            try:
                res = supabase.storage.from_(BUCKET).upload(
                    path, contents, {"content-type": file.content_type}
                )
            except Exception as e:
                logging.error(f"Upload failed for {file.filename}: {str(e)}")
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "error",
                        "detail": f"Supabase storage upload failed: {str(e)}",
                    }
                )
                continue

            file_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

            # 5) Generate embedding
            try:
                embedding = get_image_embedding(contents)
                logging.info(
                    f"Generated embedding for {file.filename}: {len(embedding)} dimensions"
                )
            except Exception as e:
                logging.error(
                    f"Embedding generation failed for {file.filename}: {str(e)}"
                )
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "error",
                        "detail": f"Embedding generation failed: {str(e)}",
                    }
                )
                continue

            # 6) Generate unique ID
            image_id = str(uuid.uuid4())

            # 7) Insert into Pinecone
            try:
                index.upsert(
                    vectors=[
                        (
                            image_id,
                            embedding,
                            {
                                "couple_id": couple_id,
                                "category": category,
                                "file_path": file_url,
                                "file_name": file.filename,
                                "type": "couple_image",
                            },
                        )
                    ]
                )
                logging.info(
                    f"Inserted {file.filename} into Pinecone with ID {image_id}"
                )
            except Exception as e:
                logging.error(f"Pinecone insert failed for {file.filename}: {str(e)}")
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "error",
                        "detail": f"Pinecone insert failed: {str(e)}",
                    }
                )
                continue

            # 8) Insert metadata into Supabase
            try:
                supabase.table("images").insert(
                    {
                        "id": image_id,
                        "couple_id": couple_id,
                        "file_name": file.filename,
                        "file_path": file_url,
                        "category": category,
                        "embedding_id": image_id,
                    }
                ).execute()
                new_row_created = True
                logging.info(f"Inserted {file.filename} metadata into Supabase")
            except Exception as e:
                logging.error(
                    f"Inserting metadata failed for {file.filename}: {str(e)}"
                )
                results.append(
                    {
                        "file_name": file.filename,
                        "status": "error",
                        "detail": f"Inserting metadata failed: {str(e)}",
                    }
                )
                continue

        # 9) Append result
        results.append(
            {
                "file_name": file.filename,
                "file_url": file_url,
                "duplicate_prevented": duplicate_prevented,
                "new_row_created": new_row_created,
                "status": "success",
            }
        )

    return {
        "results": results,
        "total_uploaded": len([r for r in results if r["status"] == "success"]),
        "total_failed": len([r for r in results if r["status"] == "error"]),
    }


@app.post("/upload-vendor-images")
async def upload_vendor_images(
    vendor_id: str = Form(...),
    vendor_type: str = Form("florist"),
    category: str = Form("florals"),
    city: str = Form("Austin"),
    state: str = Form("TX"),
    files: list[UploadFile] = File(...),
):
    """Upload vendor portfolio images and generate embeddings"""
    BUCKET = "vendor-images"
    results = []

    for file in files:
        # 1) Validate file type
        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            logging.warning(f"Rejected file with invalid type: {file.filename}")
            results.append(
                {
                    "file_name": file.filename,
                    "status": "error",
                    "detail": "File type not allowed. Use PNG, JPG, JPEG, or WEBP",
                }
            )
            continue

        # 2) Read file bytes
        contents = await file.read()

        # 3) Upload to Supabase Storage
        path = f"vendors/{vendor_type}/{vendor_id}/{category}/{uuid.uuid4()}_{file.filename}"
        try:
            res = supabase.storage.from_(BUCKET).upload(
                path, contents, {"content-type": file.content_type}
            )
        except Exception as e:
            logging.error(f"Upload failed for {file.filename}: {str(e)}")
            results.append(
                {
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Storage upload failed: {str(e)}",
                }
            )
            continue

        file_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{path}"

        # 4) Generate embedding
        try:
            embedding = get_image_embedding(contents)
            logging.info(
                f"Generated embedding for {vendor_type} image: {file.filename}"
            )
        except Exception as e:
            logging.error(f"Embedding generation failed for {file.filename}: {str(e)}")
            results.append(
                {
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Embedding generation failed: {str(e)}",
                }
            )
            continue

        # 5) Generate unique ID
        image_id = str(uuid.uuid4())

        # 6) Insert into Pinecone
        try:
            index.upsert(
                vectors=[
                    (
                        image_id,
                        embedding,
                        {
                            "vendor_id": vendor_id,
                            "vendor_type": vendor_type,
                            "category": category,
                            "city": city,
                            "state": state,
                            "file_path": file_url,
                            "file_name": file.filename,
                            "type": "vendor_image",
                        },
                    )
                ]
            )
            logging.info(
                f"Inserted {vendor_type} image into Pinecone: {file.filename} (city: {city})"
            )
        except Exception as e:
            logging.error(f"Pinecone insert failed for {file.filename}: {str(e)}")
            results.append(
                {
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Pinecone insert failed: {str(e)}",
                }
            )
            continue

        # 7) Insert metadata into Supabase
        try:
            supabase.table("vendor_images").insert(
                {
                    "id": image_id,
                    "vendor_id": vendor_id,
                    "file_name": file.filename,
                    "file_path": file_url,
                    "category": category,
                    "embedding_id": image_id,
                }
            ).execute()
            logging.info(f"Inserted {vendor_type} image metadata into Supabase")
        except Exception as e:
            logging.error(f"Database insert failed for {file.filename}: {str(e)}")
            results.append(
                {
                    "file_name": file.filename,
                    "status": "error",
                    "detail": f"Database insert failed: {str(e)}",
                }
            )
            continue

        # Success!
        results.append(
            {"file_name": file.filename, "file_url": file_url, "status": "success"}
        )

    return {
        "results": results,
        "vendor_id": vendor_id,
        "vendor_type": vendor_type,
        "category": category,
        "location": f"{city}, {state}",
        "total_uploaded": len([r for r in results if r["status"] == "success"]),
        "total_failed": len([r for r in results if r["status"] == "error"]),
    }


# ===================================
# MATCHING ENDPOINT
# ===================================
@app.post("/find-matching-vendors")
async def find_matching_vendors(
    couple_id: str = Form(...),
    vendor_type: str = Form("florist"),
    category: str = Form(None),
    city: str = Form("Austin"),
    state: str = Form("TX"),
    top_k: int = Form(10),
    min_match_score: float = Form(0.7),
):
    """Find vendors whose work matches the couple's aesthetic"""
    try:
        # 1. Get couple's image embeddings from Supabase
        query = (
            supabase.table("images")
            .select("id, embedding_id, category")
            .eq("couple_id", couple_id)
        )

        if category:
            query = query.eq("category", category)

        couple_images = query.execute()

        if not couple_images.data:
            raise HTTPException(
                status_code=404, detail=f"No images found for couple {couple_id}"
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
            raise HTTPException(
                status_code=404, detail="Could not fetch embeddings from Pinecone"
            )

        # 3. Calculate average embedding
        avg_embedding = np.mean(couple_embeddings, axis=0).tolist()
        logging.info(f"Calculated average aesthetic embedding for couple {couple_id}")

        # 4. Query Pinecone for similar vendor images
        search_filter = {
            "type": "vendor_image",
            "vendor_type": vendor_type,
            "city": city,
        }

        if category:
            search_filter["category"] = category

        results = index.query(
            vector=avg_embedding,
            top_k=top_k * 10,
            filter=search_filter,
            include_metadata=True,
        )

        if not results.matches:
            return {
                "couple_id": couple_id,
                "vendor_type": vendor_type,
                "category": category,
                "location": f"{city}, {state}",
                "matching_vendors": [],
                "message": f"No matching {vendor_type}s found. We're onboarding more {city} {vendor_type}s!",
            }

        # 5. Group results by vendor
        vendor_data = {}

        for match in results.matches:
            vendor_id = match.metadata.get("vendor_id")
            if not vendor_id or match.score < min_match_score:
                continue

            if vendor_id not in vendor_data:
                vendor_data[vendor_id] = {"scores": [], "sample_images": []}

            vendor_data[vendor_id]["scores"].append(match.score)

            if len(vendor_data[vendor_id]["sample_images"]) < 4:
                vendor_data[vendor_id]["sample_images"].append(
                    {
                        "file_path": match.metadata.get("file_path"),
                        "category": match.metadata.get("category"),
                        "similarity_score": round(match.score, 3),
                    }
                )

        # 6. Get full vendor profiles
        vendor_ids = list(vendor_data.keys())

        vendors_response = (
            supabase.table("vendors")
            .select(
                "id, business_name, tagline, description, instagram_handle, website, "
                "price_range, city, state, verified, years_in_business, vendor_type"
            )
            .in_("id", vendor_ids)
            .eq("is_active", True)
            .eq("vendor_type", vendor_type)
            .eq("city", city)
            .execute()
        )

        # 7. Get review statistics
        reviews_response = (
            supabase.table("vendor_reviews")
            .select("vendor_id, rating")
            .in_("vendor_id", vendor_ids)
            .eq("is_visible", True)
            .execute()
        )

        vendor_ratings = {}
        for review in reviews_response.data:
            vid = review["vendor_id"]
            if vid not in vendor_ratings:
                vendor_ratings[vid] = []
            vendor_ratings[vid].append(review["rating"])

        # 8. Combine scores with vendor profiles
        matching_vendors = []

        for vendor in vendors_response.data:
            vendor_id = vendor["id"]
            scores = vendor_data[vendor_id]["scores"]
            # Use top 5 scores only (or fewer if vendor has less than 5 matching images)
            top_n = 5
            top_scores = sorted(scores, reverse=True)[:top_n]
            avg_score = np.mean(top_scores)
            match_percentage = round(avg_score * 100, 1)

            ratings = vendor_ratings.get(vendor_id, [])
            avg_rating = round(np.mean(ratings), 1) if ratings else None
            review_count = len(ratings)

            matching_vendors.append(
                {
                    "match_percentage": match_percentage,
                    "match_rank": None,
                    "num_matching_images": len(scores),
                    "vendor_id": vendor_id,
                    "business_name": vendor["business_name"],
                    "vendor_type": vendor["vendor_type"],
                    "tagline": vendor.get("tagline"),
                    "description": vendor["description"],
                    "instagram_handle": vendor.get("instagram_handle"),
                    "website": vendor.get("website"),
                    "price_range": vendor.get("price_range"),
                    "location": f"{vendor['city']}, {vendor['state']}",
                    "verified": vendor.get("verified", False),
                    "years_in_business": vendor.get("years_in_business"),
                    "avg_rating": avg_rating,
                    "review_count": review_count,
                    "sample_images": vendor_data[vendor_id]["sample_images"],
                }
            )

        # 9. Sort and assign ranks
        matching_vendors.sort(key=lambda x: x["match_percentage"], reverse=True)

        for rank, vendor in enumerate(matching_vendors[:top_k], start=1):
            vendor["match_rank"] = rank

        # 10. Log match for analytics
        for vendor in matching_vendors[:top_k]:
            try:
                supabase.table("vendor_matches").insert(
                    {
                        "couple_id": couple_id,
                        "vendor_id": vendor["vendor_id"],
                        "match_score": vendor["match_percentage"] / 100,
                        "match_rank": vendor["match_rank"],
                        "matched_category": category,
                        "num_images_compared": len(couple_embeddings),
                    }
                ).execute()
            except Exception as e:
                logging.warning(f"Failed to log match: {str(e)}")

        logging.info(
            f"Found {len(matching_vendors)} matching {vendor_type}s for couple {couple_id}"
        )

        return {
            "couple_id": couple_id,
            "vendor_type": vendor_type,
            "category": category,
            "location": f"{city}, {state}",
            "num_couple_images_analyzed": len(couple_embeddings),
            "matching_vendors": matching_vendors[:top_k],
            "total_vendors_found": len(matching_vendors),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Vendor matching failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


# ===================================
# Returns vendor information for profile page
# ===================================
@app.get("/vendors/{vendor_id}")
async def get_vendor_profile(vendor_id: str):
    """Get complete vendor profile with images, reviews, and stats"""
    try:
        # 1. Get vendor basic info
        vendor_response = (
            supabase.table("vendors").select("*").eq("id", vendor_id).execute()
        )

        if not vendor_response.data or len(vendor_response.data) == 0:
            raise HTTPException(status_code=404, detail="Vendor not found")

        vendor = vendor_response.data[0]

        # Check if vendor is active
        if not vendor.get("is_active", True):
            raise HTTPException(status_code=404, detail="Vendor not found")

        # 2. Get all vendor images
        images_response = (
            supabase.table("vendor_images")
            .select("id, file_path, category, display_order, created_at")
            .eq("vendor_id", vendor_id)
            .order("display_order")
            .execute()
        )

        # Group images by category
        images_by_category = {}
        all_images = []

        for img in images_response.data:
            category = img.get("category", "other")
            if category not in images_by_category:
                images_by_category[category] = []
            images_by_category[category].append(img)
            all_images.append(img)

        # 3. Get reviews and calculate stats
        reviews_response = (
            supabase.table("vendor_reviews")
            .select("id, rating, review_text, reviewer_name, created_at")
            .eq("vendor_id", vendor_id)
            .eq("is_visible", True)
            .order("created_at", desc=True)
            .execute()
        )

        reviews = reviews_response.data

        # Calculate average rating
        avg_rating = None
        if reviews:
            ratings = [r["rating"] for r in reviews if r.get("rating")]
            if ratings:
                avg_rating = round(sum(ratings) / len(ratings), 1)

        # 4. Get match statistics (optional - how many times this vendor was matched)
        matches_response = (
            supabase.table("vendor_matches")
            .select("id, match_score, matched_category")
            .eq("vendor_id", vendor_id)
            .execute()
        )

        total_matches = len(matches_response.data) if matches_response.data else 0

        # Calculate category breakdown
        category_matches = {}
        if matches_response.data:
            for match in matches_response.data:
                cat = match.get("matched_category", "other")
                category_matches[cat] = category_matches.get(cat, 0) + 1

        # 5. Format and return response
        return {
            "vendor": {
                "id": vendor["id"],
                "business_name": vendor["business_name"],
                "vendor_type": vendor["vendor_type"],
                "tagline": vendor.get("tagline"),
                "description": vendor["description"],
                "instagram_handle": vendor.get("instagram_handle"),
                "website": vendor.get("website"),
                "email": vendor.get("email"),
                "phone": vendor.get("phone"),
                "city": vendor["city"],
                "state": vendor["state"],
                "location": f"{vendor['city']}, {vendor['state']}",
                "price_range": vendor.get("price_range"),
                "verified": vendor.get("verified", False),
                "years_in_business": vendor.get("years_in_business"),
                "created_at": vendor.get("created_at"),
            },
            "portfolio": {
                "all_images": all_images,
                "by_category": images_by_category,
                "total_count": len(all_images),
            },
            "reviews": {
                "items": reviews,
                "avg_rating": avg_rating,
                "total_count": len(reviews),
            },
            "stats": {
                "total_matches": total_matches,
                "category_matches": category_matches,
                "portfolio_size": len(all_images),
                "review_count": len(reviews),
                "avg_rating": avg_rating,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to fetch vendor profile: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch vendor profile: {str(e)}"
        )


# ===================================
# Returns couple information for dashboard
# ===================================
@app.get("/couples/{couple_id}/dashboard")
async def get_couple_dashboard(couple_id: str):
    """
    Get couple's dashboard data:
    - Profile info (name, email, wedding_date)
    - Uploaded images grouped by category
    - Recent matched vendors
    """

    try:
        # 1. Get couple profile
        couple_response = (
            supabase.table("couples").select("*").eq("id", couple_id).execute()
        )

        if not couple_response.data or len(couple_response.data) == 0:
            raise HTTPException(status_code=404, detail="Couple not found")

        couple = couple_response.data[0]

        # 2. Get all uploaded images
        images_response = (
            supabase.table("images")
            .select("*")
            .eq("couple_id", couple_id)
            .order("created_at", desc=True)
            .execute()
        )

        images = images_response.data if images_response.data else []

        # Group images by category
        images_by_category = {}
        for img in images:
            category = img.get("category", "overall")
            if category not in images_by_category:
                images_by_category[category] = []
            images_by_category[category].append(
                {
                    "id": img["id"],
                    "file_path": img["file_path"],
                    "file_name": img["file_name"],
                    "category": img["category"],
                    "created_at": img["created_at"],
                }
            )

        # 3. Get matched vendors (recent matches)
        matches_response = (
            supabase.table("vendor_matches")
            .select(
                """
            *,
            vendors:vendor_id (
                id,
                business_name,
                vendor_type,
                tagline,
                location,
                price_range,
                verified,
                years_in_business
            )
        """
            )
            .eq("couple_id", couple_id)
            .order("match_score", desc=True)
            .limit(6)
            .execute()
        )

        matched_vendors = []
        if matches_response.data:
            for match in matches_response.data:
                vendor = match.get("vendors")
                if vendor:
                    # Get one sample image for the vendor
                    vendor_img_response = (
                        supabase.table("vendor_images")
                        .select("file_path")
                        .eq("vendor_id", vendor["id"])
                        .limit(1)
                        .execute()
                    )

                    sample_image = None
                    if vendor_img_response.data and len(vendor_img_response.data) > 0:
                        sample_image = vendor_img_response.data[0]["file_path"]

                    matched_vendors.append(
                        {
                            "vendor_id": vendor["id"],
                            "business_name": vendor["business_name"],
                            "vendor_type": vendor["vendor_type"],
                            "tagline": vendor.get("tagline"),
                            "location": vendor.get("location"),
                            "price_range": vendor.get("price_range"),
                            "verified": vendor.get("verified", False),
                            "years_in_business": vendor.get("years_in_business"),
                            "match_percentage": round(match["match_score"] * 100, 1),
                            "sample_image": sample_image,
                            "matched_at": match["created_at"],
                        }
                    )

        # 4. Calculate stats
        total_images = len(images)
        categories_used = list(images_by_category.keys())
        total_matches = len(matched_vendors)

        return {
            "couple": {
                "id": couple["id"],
                "name": couple["name"],
                "email": couple["email"],
                "wedding_date": couple.get("wedding_date"),
                "location": couple.get("location"),
                "budget": couple.get("budget"),
                "guest_range": couple.get("guest_range"),
            },
            "images": {
                "all": images,
                "by_category": images_by_category,
                "total_count": total_images,
                "categories": categories_used,
            },
            "matched_vendors": matched_vendors,
            "stats": {
                "total_images": total_images,
                "total_matches": total_matches,
                "categories_count": len(categories_used),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching dashboard data: {str(e)}"
        )


# ===================================
# UTILITY ENDPOINTS
# ===================================
@app.get("/")
def root():
    """Welcome endpoint"""
    return {"message": "Welcome to Planistry Backend!"}


@app.get("/health")
async def health_check():
    """Health check with service status"""
    try:
        index.describe_index_stats()
        pinecone_healthy = True
    except:
        pinecone_healthy = False

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
        "supabase_connected": supabase_healthy,
    }
