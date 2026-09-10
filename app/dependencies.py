from fastapi import Header, HTTPException
from typing import Optional
from app.config import settings

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None or x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key