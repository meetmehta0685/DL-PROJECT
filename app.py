from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from src.health_assistant import assess_pneumonia_risk
from src.inference import PneumoniaPredictor


APP_TITLE = "🫁 Pneumonia Detection (X-Ray)"
MODEL_OPTIONS = ["resnet18", "efficientnet_b0"]
DEFAULT_MODEL_PATH = Path("checkpoints/best_model_deploy.pt")


@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str, model_name: str) -> PneumoniaPredictor:
	"""Load and cache predictor so model is not reloaded on every interaction."""
	return PneumoniaPredictor(model_path=model_path, model_name=model_name)


def _save_uploaded_file(uploaded_file) -> str:
	"""Persist uploaded image to a temp file and return path."""
	suffix = Path(uploaded_file.name).suffix or ".jpg"
	with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
		temp_file.write(uploaded_file.getbuffer())
		return temp_file.name


def _render_result_card(label: str, confidence: float, prob_normal: float, prob_pneumonia: float) -> None:
	"""Display model output metrics in a compact layout."""
	prediction_status = "🔴 Pneumonia" if label == "Pneumonia" else "🟢 Normal"

	st.subheader("Prediction")
	st.markdown(f"### {prediction_status}")

	col1, col2, col3 = st.columns(3)
	col1.metric("Confidence", f"{confidence * 100:.2f}%")
	col2.metric("Pneumonia Probability", f"{prob_pneumonia * 100:.2f}%")
	col3.metric("Normal Probability", f"{prob_normal * 100:.2f}%")


def _render_risk_guidance(prob_pneumonia: float) -> None:
	"""Display rule-based health guidance from model probability."""
	guidance = assess_pneumonia_risk(prob_pneumonia)
	risk_level = guidance["risk_level"]

	st.subheader("Risk Guidance")
	if risk_level == "High":
		st.error(f"Risk Level: {risk_level}")
	elif risk_level == "Moderate":
		st.warning(f"Risk Level: {risk_level}")
	else:
		st.info(f"Risk Level: {risk_level}")

	st.write(f"**Suggested Next Step:** {guidance['next_step']}")
	st.write(f"**Note:** {guidance['warning']}")


def _render_disclaimer() -> None:
	st.markdown("---")
	st.caption(
		"⚠️ This tool is for educational and research purposes only. "
		"It is not a medical device and must not be used for clinical diagnosis. "
		"Always consult a qualified healthcare professional."
	)


def main() -> None:
	st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")
	st.title(APP_TITLE)
	st.write("Upload a chest X-ray image to get a prediction and Grad-CAM explanation.")

	with st.sidebar:
		st.header("Settings")
		model_name = st.selectbox("Model Architecture", MODEL_OPTIONS, index=0)

		model_path_input = st.text_input(
			"Model Checkpoint Path",
			value=str(DEFAULT_MODEL_PATH),
			help="Path to a .pt checkpoint file.",
		)

		model_path = Path(model_path_input).expanduser()
		if model_path.exists():
			st.success("Checkpoint found")
		else:
			st.error("Checkpoint not found")

	uploaded_file = st.file_uploader(
		"Upload Chest X-ray",
		type=["jpg", "jpeg", "png", "bmp", "webp"],
		accept_multiple_files=False,
	)

	analyze_clicked = st.button("Analyze Image", type="primary", disabled=uploaded_file is None)

	if uploaded_file is not None:
		try:
			image = Image.open(uploaded_file).convert("RGB")
			st.image(image, caption="Uploaded X-ray", width="stretch")
		except Exception as exc:
			st.error(f"Could not read uploaded image: {exc}")
			_render_disclaimer()
			return

	if analyze_clicked:
		if not model_path.exists():
			st.error("Please provide a valid checkpoint path before running inference.")
			_render_disclaimer()
			return

		temp_image_path: Optional[str] = None
		try:
			with st.spinner("Loading model..."):
				predictor = load_predictor(str(model_path), model_name)

			temp_image_path = _save_uploaded_file(uploaded_file)

			with st.spinner("Running prediction..."):
				result = predictor.predict(temp_image_path, return_gradcam=True)

			label = result["class"]
			confidence = float(result["confidence"])
			prob_normal = float(result["probabilities"]["Normal"])
			prob_pneumonia = float(result["probabilities"]["Pneumonia"])

			_render_result_card(label, confidence, prob_normal, prob_pneumonia)
			_render_risk_guidance(prob_pneumonia)

			if result.get("gradcam_overlay") is not None:
				st.subheader("Grad-CAM Explanation")
				st.image(
					result["gradcam_overlay"],
					caption="Important regions influencing prediction",
					width="stretch",
				)
			else:
				st.warning("Grad-CAM visualization is unavailable for this prediction.")

		except Exception as exc:
			st.error(f"Inference failed: {exc}")
			st.info(
				"If this error appears after switching model architecture, "
				"ensure checkpoint and selected model match (e.g., resnet18 checkpoint with resnet18)."
			)
		finally:
			if temp_image_path is not None:
				temp_path = Path(temp_image_path)
				if temp_path.exists():
					temp_path.unlink()

	_render_disclaimer()


if __name__ == "__main__":
	main()
