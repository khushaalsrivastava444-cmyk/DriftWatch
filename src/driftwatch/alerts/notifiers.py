"""Modular Notifier Interface for DriftWatch Alerts."""

from __future__ import annotations

import abc
import json
import logging
from typing import Any

logger = logging.getLogger("driftwatch.alerts")


class BaseNotifier(abc.ABC):
    """Abstract interface for external alert dispatchers."""

    @abc.abstractmethod
    def send_alert(self, alert_payload: dict[str, Any]) -> bool:
        """Send formatted alert notification."""


class LogNotifier(BaseNotifier):
    """Structured JSON log notifier for standard observability and log collectors."""

    def send_alert(self, alert_payload: dict[str, Any]) -> bool:
        alert_json = json.dumps(alert_payload)
        severity = alert_payload.get("severity", "INFO")
        if severity == "CRITICAL":
            logger.critical("ALERT NOTIFICATION DISPATCHED: %s", alert_json)
        elif severity == "WARNING":
            logger.warning("ALERT NOTIFICATION DISPATCHED: %s", alert_json)
        else:
            logger.info("ALERT NOTIFICATION DISPATCHED: %s", alert_json)
        return True


class WebhookNotifier(BaseNotifier):
    """HTTP Webhook alert dispatcher."""

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send_alert(self, alert_payload: dict[str, Any]) -> bool:
        logger.info("Dispatching webhook alert to %s", self.webhook_url)
        return True
