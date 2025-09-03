
import os
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json, time
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
assert GEMINI_API_KEY != "", "Please set GEMINI_API_KEY (env var or .env)."

# If on Windows, set tesseract.exe path here:
TESSERACT_CMD = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

OCR_LANGS = "eng+hin"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
print("Config OK. Runs will be saved under:", os.path.abspath(RUNS_DIR))

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

genai.configure(api_key=GEMINI_API_KEY)

def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_for_ocr(img_path):
    img = cv2.imread(img_path)
    assert img is not None, f"Could not read image at {img_path}"
    img = deskew(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    den = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
    thr = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morphed

def tesseract_ocr(image_np, psm=6, oem=3, lang=OCR_LANGS):
    config = f"--oem {oem} --psm {psm}"
    text = pytesseract.image_to_string(image_np, lang=lang, config=config)
    return text

def run_gemini_repair(image_path, raw_text, prompt_mode="clean"):
    """
    prompt_mode:
      - "clean": produce clean reconstructed text using both inputs.
      - "extract": return structured fields (entities) as JSON.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    base_system_prompt = (
        "You are an expert OCR post-processor. "
        "You are given (1) an original document image and (2) noisy OCR text from Tesseract.\n"
        "Task: reconstruct the most accurate text you can. "
        "Preserve reading order, headers, tables (as Markdown), and line breaks. "
        "If handwriting is present, read from the image to fill missing/incorrect words. "
        "Do not invent content that is not visible in the image."
    )
    extract_system_prompt = (
        "Extract key fields from the document. Respond with strict JSON only. "
        "For Aadhaar cards, extract 'aadhaar_number', 'full_name', 'date_of_birth', 'gender', 'address'. "
        "For Passports, extract 'passport_number', 'full_name', 'date_of_birth', 'place_of_birth', 'date_of_issue', 'date_of_expiry', 'nationality', 'gender', 'passport_type'. "
        "For other documents, use general keys like 'names', 'ids', 'dates', 'addresses', 'emails', 'phones', 'stamps_or_seals'. "
        "When unknown, use null or empty arrays. Do not include any commentary."
    )
    user_prompt = extract_system_prompt if prompt_mode == "extract" else base_system_prompt
    img = Image.open(image_path)
    resp = model.generate_content([
        {"text": user_prompt},
        img,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    return resp.text.strip()

def save_run(run_dir, settings, raw_text, gemini_text, entities_json_str=None):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "raw_ocr.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text or "")
    with open(os.path.join(run_dir, "gemini_text.txt"), "w", encoding="utf-8") as f:
        f.write(gemini_text or "")
    if entities_json_str:
        with open(os.path.join(run_dir, "entities.json"), "w", encoding="utf-8") as f:
            f.write(entities_json_str)
    print("Saved to:", os.path.abspath(run_dir))
    return os.path.abspath(run_dir)

def classify_document(image_path, raw_text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "Analyze the provided document image and its OCR text. "
        "Classify the document as 'Aadhaar', 'Passport', or 'Other'. "
        "Respond with only one of these three words."
    )
    img = Image.open(image_path)
    resp = model.generate_content([
        {"text": prompt},
        img,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    return resp.text.strip()


import datetime

def check_expiry(date_string):
    if not date_string:
        return "No date provided."
    try:
        # Attempt to parse the date in common formats
        fmts = ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y", "%Y.%m.%d", "%d.%m.%Y"]
        parsed_date = None
        for fmt in fmts:
            try:
                parsed_date = datetime.datetime.strptime(date_string, fmt).date()
                break
            except ValueError:
                continue

        if not parsed_date:
            return f"Could not parse date: {date_string}"

        today = datetime.date.today()
        if parsed_date < today:
            return "Expired"
        else:
            return "Valid"
    except Exception as e:
        return f"Error checking expiry for {date_string}: {e}"


import os
import time

image_path = input("Please enter the path to your image file: ")

if not image_path:
    raise ValueError("No image path provided!")

print(f"Selected file: {image_path}")

# Run your pipeline
pre = preprocess_for_ocr(image_path)
raw_text = tesseract_ocr(pre, psm=6, oem=3)

document_type = classify_document(image_path, raw_text)
print(f"Document classified as: {document_type}")

gemini_text = run_gemini_repair(image_path, raw_text, prompt_mode="clean")

DO_EXTRACT = True
entities_json_str = None
if DO_EXTRACT:
    entities_json_str = run_gemini_repair(image_path, raw_text, prompt_mode="extract")

ts = time.strftime("%Y%m%d-%H%M%S")
run_dir = os.path.join(RUNS_DIR, f"run-{ts}")
settings = {
    "languages": OCR_LANGS,
    "tesseract_cmd": TESSERACT_CMD,
    "psm": 6,
    "oem": 3,
    "model": "gemini-2.5-flash",
    "prompt_modes": ["clean", "extract" if DO_EXTRACT else "clean"],
    "image_path": image_path,
}
save_run(run_dir, settings, raw_text, gemini_text, entities_json_str)

print("\n--- RAW OCR (first 500 chars) ---\n", (raw_text or "")[:500])
print("\n--- GEMINI CLEAN TEXT (first 500 chars) ---\n", (gemini_text or "")[:500])
if entities_json_str:
    print("\n--- ENTITIES JSON (first 500 chars) ---\n", entities_json_str[:500])
    try:
        entities = json.loads(entities_json_str)
        if document_type == "Passport":
            date_of_expiry = entities.get("date_of_expiry")
            if date_of_expiry:
                expiry_status = check_expiry(date_of_expiry)
                print(f"Passport Expiry Status ({date_of_expiry}): {expiry_status}")
        # You can add similar checks for other date fields if needed

    except json.JSONDecodeError:
        print("Error: Could not decode entities JSON.")
