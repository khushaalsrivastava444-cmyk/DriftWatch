"""Database Initialization and Data Retention Management."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from driftwatch.db.base import Base, engine
from driftwatch.db.models import PredictionLog

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Create all tables in the database if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


def apply_retention_policy(db: Session, retention_days: int = 90) -> int:
    """Prune prediction logs older than retention_days.
    
    Returns:
        Number of pruned prediction log rows.
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    stmt = delete(PredictionLog).where(PredictionLog.timestamp < cutoff)
    result = db.execute(stmt)
    db.commit()
    pruned_count = result.rowcount or 0
    logger.info("Applied retention policy: pruned %d prediction logs older than %s", pruned_count, cutoff)
    return pruned_count
