"""Authentication and Authorization Middleware for Write and Admin Endpoints."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

DEFAULT_API_KEY = os.getenv("DRIFTWATCH_API_KEY", "driftwatch-secret-key")


def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> str:
    """Verify write/admin API key from X-API-Key header or Authorization Bearer header.
    
    Raises:
        HTTPException 401 if missing or invalid.
    """
    key = x_api_key
    if not key and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            key = parts[1]

    if not key or key != DEFAULT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "UNAUTHORIZED",
                "message": "Valid API key is required in 'X-API-Key' or 'Authorization: Bearer <key>' header.",
            },
        )
    return key
