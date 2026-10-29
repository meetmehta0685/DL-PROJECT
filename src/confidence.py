"""
Map the model's pneumonia probability to a descriptive confidence band.
The probabilities are not calibrated, so the band describes model output only.
"""

from typing import Dict


def confidence_band(prob_pneumonia: float) -> Dict[str, str]:
    """
    Describe how strongly the model leans towards pneumonia.

    Args:
        prob_pneumonia (float): Model probability for pneumonia in [0, 1]

    Returns:
        Dict[str, str]: band, description
    """
    if prob_pneumonia >= 0.85:
        return {"band": "Strong pneumonia signal", "description": "The model strongly favours the Pneumonia class."}
    if prob_pneumonia >= 0.60:
        return {"band": "Moderate pneumonia signal", "description": "The model leans towards Pneumonia."}
    if prob_pneumonia >= 0.40:
        return {"band": "Uncertain", "description": "The model cannot clearly separate the two classes for this image."}
    return {"band": "Normal signal", "description": "The model favours the Normal class."}
