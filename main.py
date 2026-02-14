import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")

# Initialize Gemini Model
# Using gemini-2.0-flash as 1.5 is not available for this key/region
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(title="LocalMind AI Backend")

# CORS - Allow all for MVP/Hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LocalMind AI"}

@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    location: Optional[str] = Form(None), # Expected format: "lat,long" or "City, Area"
    language_code: str = Form("en") # e.g., 'hi', 'en', 'te'
):
    """
    Multimodal analysis endpoint.
    Accepts Text (optional), Image (optional), Location (required for best results).
    """
    try:
        if not API_KEY:
             raise HTTPException(status_code=500, detail="Server misconfiguration: API Key missing.")
        
        if not text and not image:
            raise HTTPException(status_code=400, detail="Either text or image must be provided.")

        inputs = []
        
        # 1. System Prompt / Context Construction
        base_prompt = """
        You are LocalMind AI, a hyperlocal assistant for India.
        Your goal is to provide real-time, actionable intelligence based on the user's location and input.
        
        CONTEXT:
        - Location: {location}
        - User's Language Preference: {language_code} (Respond in this language OR English if unsure, but prefer mixed/colloquial if appropriate like Hinglish).
        
        INSTRUCTIONS:
        1. Analyze the input (image and/or text).
        2. Identify specific local details (shops, signs, food, transport, safety).
        3. Provide estimated prices, safety tips, or transport options if relevant.
        4. Be CONCISE and ACTIONABLE. No long wiki-style answers.
        5. If the user asks about a price (e.g., auto rickshaw), give a realistic estimate for that Indian city.
        
        USER INPUT:
        {user_text}
        """
        
        formatted_prompt = base_prompt.format(
            location=location or "Unknown India Location",
            language_code=language_code,
            user_text=text or "Analyze this image."
        )
        
        inputs.append(formatted_prompt)

        # 2. Process Image
        if image:
            content = await image.read()
            image_part = {"mime_type": image.content_type, "data": content}
            # Or use PIL to verify it's an image
            try:
                img = Image.open(io.BytesIO(content))
                inputs.append(img)
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                raise HTTPException(status_code=400, detail="Invalid image file.")

        # 3. Call Gemini
        logger.info(f"Sending request to Gemini... Location: {location}")
        response = model.generate_content(inputs)
        
        response_text = response.text
        
        # 4. Return formatted response
        return JSONResponse(content={
            "response": response_text,
            "location_context": location
        })

    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
