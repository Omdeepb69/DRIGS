"""Control Plane REST API Security & Bearer Token / API Key Authentication."""

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional
from fastapi import HTTPException, Request, status


def create_hmac_token(api_key: str, timestamp: Optional[float] = None) -> str:
    """Generate a time-stamped HMAC Bearer Token using an API key secret."""
    if timestamp is None:
        timestamp = time.time()
    ts_str = str(int(timestamp))
    signature = hmac.new(
        api_key.encode("utf-8"),
        ts_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{ts_str}.{signature}"


def verify_bearer_token(token: str, expected_api_key: str) -> bool:
    """Verify raw API key or HMAC signed Bearer token against expected secret."""
    if not expected_api_key:
        return True
    if not token:
        return False
    # Direct static API key comparison using constant-time comparison
    if secrets.compare_digest(token, expected_api_key):
        return True
    # HMAC signature match: timestamp.signature or timestamp:signature
    try:
        if "." in token or ":" in token:
            sep = "." if "." in token else ":"
            ts_str, signature = token.rsplit(sep, 1)
            expected_sig = hmac.new(
                expected_api_key.encode("utf-8"),
                ts_str.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if secrets.compare_digest(signature, expected_sig):
                ts = float(ts_str)
                # Verify token freshness (24-hour validity window)
                if abs(time.time() - ts) < 86400:
                    return True
    except Exception:
        pass
    return False


class APIKeyAuth:
    """FastAPI authentication dependency enforcing API Key / Bearer token security."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.getenv("DRIGS_API_KEY")

    def __call__(self, request: Request) -> Optional[str]:
        if not self.api_key:
            return None  # Unauthenticated / local dev mode

        # Health check endpoint is exempt from authentication
        if request.url.path == "/v1/health":
            return None

        token: Optional[str] = None

        auth_header = request.headers.get("Authorization")
        if auth_header:
            if auth_header.startswith("Bearer "):
                token = auth_header[7:].strip()
            elif auth_header.startswith("bearer "):
                token = auth_header[7:].strip()
            else:
                token = auth_header.strip()

        if not token:
            token = request.headers.get("X-API-Key")

        if not token or not verify_bearer_token(token, self.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key / Bearer Token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return token
