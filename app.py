from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from src.confidence import confidence_band
from src.image_processing import colorize_heatmap, run_pipeline
from src.inference import PneumoniaPredictor


APP_TITLE = "🫁 Pneumonia Detection (X-Ray)"
MODEL_NAME = "resnet18"
MODEL_PATH = Path("checkpoints/best_model_deploy.pt")
SAMPLES_DIR = Path("samples")
SAMPLES = {
	"Normal 1": SAMPLES_DIR / "normal_1.jpeg",
	"Normal 2": SAMPLES_DIR / "normal_2.jpeg",
	"Pneumonia (bacterial)": SAMPLES_DIR / "pneumonia_bacterial.jpeg",
	"Pneumonia (viral)": SAMPLES_DIR / "pneumonia_viral.jpeg",
}
# Results of best_model_deploy.pt on the 624-image test split (report/.audit/evaluation.json)
TEST_METRICS = {"Accuracy": "87.0%", "Precision": "89.3%", "Recall": "90.0%", "F1-score": "89.7%"}


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


def _render_sidebar() -> None:
	"""Show model details and test-set results."""
	with st.sidebar:
		st.header("Model")
		st.markdown(
			f"**ResNet18** (ImageNet transfer learning)  \n"
			f"Input: 224×224 X-ray  \n"
			f"Output: Normal / Pneumonia  \n"
			f"Explainability: Grad-CAM on `layer4`"
		)
		st.subheader("Test-set results")
		st.caption("624 unseen test images")
		col1, col2 = st.columns(2)
		for i, (name, value) in enumerate(TEST_METRICS.items()):
			(col1 if i % 2 == 0 else col2).metric(name, value)

		if not MODEL_PATH.exists():
			st.error(f"Model checkpoint not found: {MODEL_PATH}")


def _select_image() -> tuple[Optional[Image.Image], Optional[str], object]:
	"""Return (image, label, source) where source is the uploaded file or a sample path."""
	if "uploader_key" not in st.session_state:
		st.session_state.uploader_key = 0
	if "sample" not in st.session_state:
		st.session_state.sample = None

	uploaded_file = st.file_uploader(
		"Upload Chest X-ray",
		type=["jpg", "jpeg", "png", "bmp", "webp"],
		accept_multiple_files=False,
		key=f"uploader_{st.session_state.uploader_key}",
	)

	st.write("Or try a sample X-ray from the test set:")
	sample_cols = st.columns(len(SAMPLES))
	for col, (label, path) in zip(sample_cols, SAMPLES.items()):
		if col.button(label, width="stretch"):
			st.session_state.sample = label
			# A new uploader key clears any uploaded file so the sample is used
			st.session_state.uploader_key += 1
			st.rerun()

	if uploaded_file is not None:
		st.session_state.sample = None
		try:
			image = Image.open(uploaded_file).convert("RGB")
		except Exception as exc:
			st.error(f"Could not read uploaded image: {exc}")
			return None, None, None
		return image, uploaded_file.name, uploaded_file

	if st.session_state.sample is not None:
		path = SAMPLES[st.session_state.sample]
		return Image.open(path).convert("RGB"), f"Sample: {st.session_state.sample}", path

	return None, None, None


def _render_histogram(before: np.ndarray, after: np.ndarray):
	"""Plot intensity histograms before and after CLAHE."""
	fig, ax = plt.subplots(figsize=(4, 3))
	ax.hist(before.ravel(), bins=64, range=(0, 255), alpha=0.6, label="Resized")
	ax.hist(after.ravel(), bins=64, range=(0, 255), alpha=0.6, label="After CLAHE")
	ax.set_xlabel("Pixel intensity")
	ax.set_ylabel("Pixel count")
	ax.legend(fontsize=8)
	fig.tight_layout()
	return fig


def _render_pipeline(image: Image.Image) -> None:
	"""Show each image processing step applied to the X-ray."""
	steps = run_pipeline(image)
	stats = steps["stats"]

	with st.expander("🔬 Image Processing Pipeline", expanded=True):
		st.caption(
			"The model receives steps 2, 3 and 6. CLAHE, edge detection and the histogram "
			"are shown to analyse the image and are not fed to the model."
		)

		row1 = st.columns(3)
		row1[0].image(steps["original"], caption=f"Step 1: Original ({stats['original_size']} px)", width="stretch")
		row1[1].image(steps["gray"], caption="Step 2: Grayscale (1 channel, copied to RGB for the CNN)", width="stretch")
		row1[2].image(steps["resized"], caption="Step 3: Resized to 224×224 (bilinear)", width="stretch")

		row2 = st.columns(3)
		row2[0].image(
			steps["clahe"],
			caption=f"Step 4: CLAHE contrast enhancement (std {stats['contrast_before']:.1f} → {stats['contrast_after']:.1f})",
			width="stretch",
		)
		row2[1].image(
			steps["edges"],
			caption=f"Step 5: Gaussian blur + Canny edges ({stats['edge_density']:.1%} edge pixels)",
			width="stretch",
		)
		row2[2].image(
			steps["normalized"],
			caption=(
				f"Step 6: Normalised input (x − mean) / std, range "
				f"{stats['normalized_min']:.2f} to {stats['normalized_max']:.2f}"
			),
			width="stretch",
		)

		hist_col, text_col = st.columns([1, 1])
		with hist_col:
			fig = _render_histogram(steps["resized"], steps["clahe"])
			st.pyplot(fig)
			plt.close(fig)
		with text_col:
			st.markdown(
				f"**Intensity histogram**  \n"
				f"Mean intensity: {stats['mean_intensity']:.1f} / 255  \n"
				f"CLAHE equalises the histogram locally in 8×8 tiles, which makes "
				f"opacities in the lung fields easier to see."
			)


def _render_result(label: str, confidence: float, prob_normal: float, prob_pneumonia: float) -> None:
	"""Display model output metrics and the confidence band."""
	prediction_status = "🔴 Pneumonia" if label == "Pneumonia" else "🟢 Normal"

	st.subheader("Prediction")
	st.markdown(f"### {prediction_status}")

	col1, col2, col3 = st.columns(3)
	col1.metric("Confidence", f"{confidence * 100:.2f}%")
	col2.metric("Pneumonia Probability", f"{prob_pneumonia * 100:.2f}%")
	col3.metric("Normal Probability", f"{prob_normal * 100:.2f}%")

	band = confidence_band(prob_pneumonia)
	st.info(f"**Model confidence band:** {band['band']}. {band['description']}")


def _render_gradcam(image: Image.Image, result: dict) -> None:
	"""Show original, Grad-CAM heatmap and overlay side by side."""
	st.subheader("Grad-CAM Explanation")
	if result.get("gradcam_overlay") is None:
		st.warning("Grad-CAM visualization is unavailable for this prediction.")
		return

	resized = np.array(image.resize((224, 224), Image.BILINEAR))
	cols = st.columns(3)
	cols[0].image(resized, caption="Input (224×224)", width="stretch")
	cols[1].image(colorize_heatmap(result["gradcam_heatmap"]), caption="Grad-CAM heatmap", width="stretch")
	cols[2].image(result["gradcam_overlay"], caption="Overlay (60% image + 40% heatmap)", width="stretch")
	st.caption(
		"Red regions influenced the prediction most. Grad-CAM is computed on a 7×7 feature map and "
		"upsampled, so it is coarse: it shows where the model looked, not a lesion outline."
	)


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
	st.write(
		"Upload a chest X-ray to see each image processing step, the model's prediction "
		"and a Grad-CAM explanation."
	)

	_render_sidebar()

	image, source_label, source = _select_image()
	if image is None:
		_render_disclaimer()
		return

	st.caption(f"Current image: {source_label}")
	_render_pipeline(image)

	analyze_clicked = st.button("Analyze Image", type="primary")
	if analyze_clicked:
		if not MODEL_PATH.exists():
			st.error("Model checkpoint is missing, so inference cannot run.")
			_render_disclaimer()
			return

		temp_image_path: Optional[str] = None
		try:
			with st.spinner("Loading model..."):
				predictor = load_predictor(str(MODEL_PATH), MODEL_NAME)

			if isinstance(source, Path):
				image_path = str(source)
			else:
				temp_image_path = _save_uploaded_file(source)
				image_path = temp_image_path

			with st.spinner("Running prediction..."):
				result = predictor.predict(image_path, return_gradcam=True)

			_render_result(
				result["class"],
				float(result["confidence"]),
				float(result["probabilities"]["Normal"]),
				float(result["probabilities"]["Pneumonia"]),
			)
			_render_gradcam(image, result)

		except Exception as exc:
			st.error(f"Inference failed: {exc}")
		finally:
			if temp_image_path is not None:
				temp_path = Path(temp_image_path)
				if temp_path.exists():
					temp_path.unlink()

	_render_disclaimer()


if __name__ == "__main__":
	main()
