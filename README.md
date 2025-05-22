# Pneumonia X-ray Detection (PyTorch + Streamlit)

A simple web app to detect pneumonia from chest X-ray images using a CNN model built in PyTorch and deployed with Streamlit.

🔗 **Live App**: [Click to Open](https://pneumonia-xray-detection-qjga3lwjs6x4htk4heuyni.streamlit.app)

⚠️ **Note**: This system is trained only to classify chest X-ray images as either `NORMAL` or `PNEUMONIA`. It may incorrectly classify unrelated or non-X-ray images, as well as other diseases, as `PNEUMONIA`, because it was not trained to recognize other categories or conditions.

---

## 📦 How to Use (Local Setup)

1. Clone the repository:

```bash
git clone https://github.com/TodEscbr/pneumonia-xray-detection.git
cd pneumonia-xray-detection/web_app
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

The model will be automatically downloaded from Google Drive on first run.

---

## 🛠 Tech Used

* PyTorch
* Streamlit
* gdown

---

## 👤 Author

Muhammad Arsyad Bin Mansor

---

📍 For educational use only — not for medical diagnosis.
