"""SQLAlchemy Database Configuration and Session Management."""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DRIFTWATCH_DB_URL", "sqlite:///driftwatch.db")

# SQLite needs check_same_thread=False for multi-threaded FastAPI
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base class for all DriftWatch database models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding database session with automatic teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
