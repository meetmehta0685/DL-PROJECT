"""
Rule-based health assistant outputs for pneumonia risk.
"""

from typing import Dict


def assess_pneumonia_risk(prob_pneumonia: float) -> Dict[str, str]:
    """
    Map pneumonia probability to risk level, next step, and warning message.

    Args:
        prob_pneumonia (float): Model probability for pneumonia in [0, 1]

    Returns:
        Dict[str, str]: risk_level, next_step, warning
    """
    if prob_pneumonia >= 0.85:
        return {
            "risk_level": "High",
            "next_step": "Consult a physician immediately",
            "warning": "High risk - seek urgent medical evaluation."
        }
    if prob_pneumonia >= 0.60:
        return {
            "risk_level": "Moderate",
            "next_step": "Schedule a medical review soon",
            "warning": "Moderate risk - monitor symptoms and follow up."
        }
    if prob_pneumonia >= 0.40:
        return {
            "risk_level": "Low",
            "next_step": "Monitor and consider retest if symptoms persist",
            "warning": "Low risk - continue observation."
        }
    return {
        "risk_level": "Minimal",
        "next_step": "No immediate action required",
        "warning": "Minimal risk - maintain routine care."
    }
