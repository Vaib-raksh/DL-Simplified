"""Configurable temporal rules for cautious drowsiness demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

from .features import FacialFeatures


@dataclass(frozen=True)
class RuleConfig:
    """Provisional thresholds and durations for the rule baseline."""

    ear_threshold: float = 0.21
    mar_threshold: float = 0.60
    closure_duration_seconds: float = 1.5
    mouth_duration_seconds: float = 1.0
    alert_cooldown_seconds: float = 5.0


@dataclass(frozen=True)
class RuleDecision:
    """Human-readable rule output for one frame."""

    status: str
    reason: str
    alert: bool = False


class TemporalRuleBaseline:
    """Track sustained conditions using elapsed timestamps, not frame counts."""

    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()
        self._closed_since: int | None = None
        self._mouth_open_since: int | None = None
        self._last_alert_at: int | None = None

    def reset(self) -> None:
        """Invalidate temporal state after a tracking gap."""
        self._closed_since = None
        self._mouth_open_since = None

    def _can_alert(self, timestamp_ms: int) -> bool:
        if (
            self._last_alert_at is not None
            and timestamp_ms - self._last_alert_at
            < self.config.alert_cooldown_seconds * 1000
        ):
            return False
        self._last_alert_at = timestamp_ms
        return True

    def update(
        self, features: FacialFeatures | None, timestamp_ms: int
    ) -> RuleDecision:
        """Return a decision and reset state when measurements are unreliable."""
        timestamp_ms = max(0, int(timestamp_ms))
        if features is None or not features.valid or features.average_ear is None:
            self.reset()
            reason = features.reason if features else "no measurements"
            return RuleDecision("UNABLE TO ASSESS", f"Tracking unavailable: {reason}.")

        eyes_closed = features.average_ear < self.config.ear_threshold
        mouth_open = features.mar >= self.config.mar_threshold
        if eyes_closed:
            self._mouth_open_since = None
            if self._closed_since is None:
                self._closed_since = timestamp_ms
            duration = (timestamp_ms - self._closed_since) / 1000
            if duration >= self.config.closure_duration_seconds:
                return RuleDecision(
                    "SUSTAINED EYE CLOSURE",
                    "Eyes remain below the provisional EAR threshold; this is a rule alert, not a diagnosis.",
                    self._can_alert(timestamp_ms),
                )
            return RuleDecision(
                "POSSIBLE BLINK",
                "Eyes are below the provisional EAR threshold; waiting for sustained duration.",
            )

        self._closed_since = None
        if mouth_open:
            if self._mouth_open_since is None:
                self._mouth_open_since = timestamp_ms
            duration = (timestamp_ms - self._mouth_open_since) / 1000
            if duration >= self.config.mouth_duration_seconds:
                return RuleDecision(
                    "SUSTAINED MOUTH OPENING",
                    "Mouth remains open; this may be yawning, but mouth opening alone is not confirmed yawning.",
                    self._can_alert(timestamp_ms),
                )
            return RuleDecision(
                "MOUTH OPENING",
                "Mouth is above the provisional MAR threshold; waiting for sustained duration.",
            )

        self._mouth_open_since = None
        return RuleDecision("NORMAL TRACKING", "No sustained rule condition detected.")
