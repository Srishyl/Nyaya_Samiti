import streamlit as st
import os
from dotenv import load_dotenv
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json, time
import google.generativeai as genai
import datetime
import re # Added for regex operations
import torch
import io # Added for BytesIO
from model.object_detection.faster_rcnn import StampSealDetector
from model.signature_verification.siamese_network import SiameseNetwork
from torchvision import transforms
import torch.nn.functional as F # Added for F.pairwise_distance
from model.object_detection.embedding_matcher import EmbeddingMatcher
from model.signature_verification.docsignaturenet import DocSignatureNet
from model.tamper_detection.mantra_net import ManTraNet
from model.tamper_detection.exif_cv_hybrid import EXIF_CV_Hybrid
from model.tamper_detection.vision_transformers import VisionTransformerTamperDetector
from model.tamper_detection.autoencoder_gan import Autoencoder, Discriminator

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
assert GEMINI_API_KEY != "", "Please set GEMINI_API_KEY (env var or .env)."

# If on Windows, set tesseract.exe path here:
TESSERACT_CMD = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

OCR_LANGS = "eng+hin"
RUNS_DIR = "runs"
OBJECT_DETECTION_OUTPUT_DIR = os.path.join(RUNS_DIR, "object_detection")
SIGNATURE_VERIFICATION_DIR = os.path.join(RUNS_DIR, "signature_verification")
EMBEDDING_MATCHING_DIR = os.path.join(RUNS_DIR, "embedding_matching")
DOCSIGNATURENET_DIR = os.path.join(RUNS_DIR, "docsignaturenet")
TAMPER_DETECTION_MANTRA_DIR = os.path.join(RUNS_DIR, "tamper_detection_mantra")
TAMPER_DETECTION_EXIF_CV_DIR = os.path.join(RUNS_DIR, "tamper_detection_exif_cv")
TAMPER_DETECTION_VIT_DIR = os.path.join(RUNS_DIR, "tamper_detection_vit")
TAMPER_DETECTION_AUTO_GAN_DIR = os.path.join(RUNS_DIR, "tamper_detection_auto_gan")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(OBJECT_DETECTION_OUTPUT_DIR, exist_ok=True)
os.makedirs(SIGNATURE_VERIFICATION_DIR, exist_ok=True)
os.makedirs(EMBEDDING_MATCHING_DIR, exist_ok=True)
os.makedirs(DOCSIGNATURENET_DIR, exist_ok=True)
os.makedirs(TAMPER_DETECTION_MANTRA_DIR, exist_ok=True)
os.makedirs(TAMPER_DETECTION_EXIF_CV_DIR, exist_ok=True)
os.makedirs(TAMPER_DETECTION_VIT_DIR, exist_ok=True)
os.makedirs(TAMPER_DETECTION_AUTO_GAN_DIR, exist_ok=True)

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

def preprocess_for_ocr(img_np):
    img = img_np
    assert img is not None, f"Could not read image"
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

def run_gemini_repair(image_pil, raw_text, prompt_mode="clean"):
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
    resp = model.generate_content([
        {"text": user_prompt},
        image_pil,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    full_response_text = resp.text.strip()

    if prompt_mode == "extract":
        # Use regex to find and extract the JSON object
        json_match = re.search(r'```json\n({.*?})\n```', full_response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        else:
            # If no code block, try to find a standalone JSON object
            json_match = re.search(r'{.*}', full_response_text, re.DOTALL)
            if json_match:
                return json_match.group(0)
            else:
                return "{}" # Return empty JSON if no JSON found
    return full_response_text

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
    # print("Saved to:", os.path.abspath(run_dir)) # Suppress console output for Streamlit
    return os.path.abspath(run_dir)

def classify_document(image_pil, raw_text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "Analyze the provided document image and its OCR text. "
        "Classify the document as 'Aadhaar', 'Passport', or 'Other'. "
        "Respond with only one of these three words."
    )
    resp = model.generate_content([
        {"text": prompt},
        image_pil,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    return resp.text.strip()

def check_expiry(date_string, is_expiry_date=True):
    if not date_string:
        return "No date provided."
    try:
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
            return "Expired" if is_expiry_date else "Valid (Past Date)"
        else:
            return "Valid"
    except Exception as e:
        return f"Error checking date for {date_string}: {e}"

def run_gemini_visual_analysis(image_pil, raw_text):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = (
        "Analyze the provided document image and its OCR text for visual anomalies. "
        "Specifically, look for signs of: "
        "1. **Stamp/Seal/Signature Presence/Anomalies:** Is there a stamp, seal, or signature? Does it appear authentic or show any irregularities (e.g., pixelation, misalignment)? "
        "2. **Tamper Detection:** Are there any visual cues suggesting the image has been altered or forged (e.g., inconsistent lighting, mismatched fonts, cut-and-paste artifacts, unusual image compression)? "
        "Provide a brief assessment for each point. State clearly if no anomalies are detected or if a feature is not present."
    )
    resp = model.generate_content([
        {"text": prompt},
        image_pil,
        {"text": "-----\nOCR (Tesseract) text:\n" + str(raw_text) + "\n-----\n"}
    ])
    return resp.text.strip()

st.title("Document OCR and Validation")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    # temp_image_path = os.path.join(RUNS_DIR, uploaded_file.name)
    # with open(temp_image_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())

    # Read image into PIL Image object and NumPy array for in-memory processing
    img_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    opencv_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    st.image(uploaded_file, caption="Uploaded Image.", width='stretch')
    st.write("")
    st.write("Processing...")

    # Run your pipeline
    pre = preprocess_for_ocr(opencv_image)
    raw_text = tesseract_ocr(pre, psm=6, oem=3)

    document_type = classify_document(pil_image, raw_text)
    st.subheader(f"Document Classified as: {document_type}")

    gemini_text = run_gemini_repair(pil_image, raw_text, prompt_mode="clean")
    st.subheader("Gemini Cleaned Text:")
    st.write(gemini_text[:500] + "...") # Display first 500 chars

    # Initialize StampSealDetector
    stamp_seal_detector = StampSealDetector(num_classes=2) # 1 class (stamp/seal) + background

    st.subheader("Stamp/Seal Detection:")
    if st.button("Run Stamp/Seal Detection"):
        with st.spinner("Detecting stamps/seals..."):
            detections = stamp_seal_detector.detect(pil_image)
            if detections:
                st.success("Stamps/Seals detected!")
                output_detection_path = os.path.join(OBJECT_DETECTION_OUTPUT_DIR, "detected_stamps_seals.jpg")
                stamp_seal_detector.visualize_detections(pil_image, detections, output_detection_path)
                st.image(output_detection_path, caption="Detected Stamps/Seals.", width='stretch')
                st.json(detections)
            else:
                st.info("No stamps or seals detected.")

    # Initialize Siamese Network
    siamese_net = SiameseNetwork()
    siamese_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)), # Assuming 100x100 input for SiameseNetwork
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    st.subheader("Signature Verification (Siamese Network):")
    st.warning("Note: This is a placeholder. For real-world use, you would train the Siamese Network on a dedicated signature dataset and load pre-trained weights.")

    uploaded_reference_signature = st.file_uploader("Upload Reference Signature Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="ref_sig")
    uploaded_new_signature = st.file_uploader("Upload New Signature Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="new_sig")

    if uploaded_reference_signature and uploaded_new_signature:
        with st.spinner("Verifying signatures..."):
            # Save uploaded files temporarily
            ref_sig_path = os.path.join(SIGNATURE_VERIFICATION_DIR, "reference_signature.png")
            new_sig_path = os.path.join(SIGNATURE_VERIFICATION_DIR, "new_signature.png")
            with open(ref_sig_path, "wb") as f: f.write(uploaded_reference_signature.getbuffer())
            with open(new_sig_path, "wb") as f: f.write(uploaded_new_signature.getbuffer())

            img_ref = Image.open(ref_sig_path).convert("RGB")
            img_new = Image.open(new_sig_path).convert("RGB")

            st.image([img_ref, img_new], caption=["Reference Signature", "New Signature"], width=150)

            # Convert to grayscale and apply transform
            img_ref_tensor = siamese_transform(img_ref).unsqueeze(0)
            img_new_tensor = siamese_transform(img_new).unsqueeze(0)

            output1, output2 = siamese_net(img_ref_tensor, img_new_tensor)
            euclidean_distance = F.pairwise_distance(output1, output2)

            similarity_threshold = 1.0 # This threshold needs to be determined by training
            if euclidean_distance.item() < similarity_threshold:
                st.success(f"Signatures are LIKELY GENUINE! Distance: {euclidean_distance.item():.4f}")
            else:
                st.error(f"Signatures are LIKELY FORGED! Distance: {euclidean_distance.item():.4f}")

            os.remove(ref_sig_path)
            os.remove(new_sig_path)

    # Initialize EmbeddingMatcher
    embedding_matcher = EmbeddingMatcher(model_name='resnet18') # Can be 'efficientnet_b0'

    st.subheader("Stamp/Seal Embedding Matching:")
    st.warning("Note: This is a placeholder. For real-world use, you would have a database of known stamp/seal embeddings for comparison.")

    uploaded_image_for_embedding_1 = st.file_uploader("Upload Image 1 for Embedding Matching (PNG, JPG)", type=["png", "jpg", "jpeg"], key="emb_img1")
    uploaded_image_for_embedding_2 = st.file_uploader("Upload Image 2 for Embedding Matching (PNG, JPG)", type=["png", "jpg", "jpeg"], key="emb_img2")

    if uploaded_image_for_embedding_1 and uploaded_image_for_embedding_2:
        with st.spinner("Matching embeddings..."):
            # Read uploaded files into PIL Image objects
            img_emb1 = Image.open(io.BytesIO(uploaded_image_for_embedding_1.getvalue())).convert("RGB")
            img_emb2 = Image.open(io.BytesIO(uploaded_image_for_embedding_2.getvalue())).convert("RGB")

            st.image([img_emb1, img_emb2], caption=["Image 1", "Image 2"], width=150)

            embedding1 = embedding_matcher.get_embedding(img_emb1)
            embedding2 = embedding_matcher.get_embedding(img_emb2)

            similarity, is_match = embedding_matcher.match_embeddings(embedding1, embedding2)

            st.write(f"Similarity Score: {similarity:.4f}")
            if is_match:
                st.success("Images are LIKELY A MATCH (Cosine Similarity > 0.8)")
            else:
                st.info("Images are LIKELY NOT A MATCH (Cosine Similarity <= 0.8)")

    # Initialize DocSignatureNet
    docsignature_net = DocSignatureNet(num_classes=2)
    docsignature_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((150, 220)), # Assuming 150x220 input for DocSignatureNet
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    st.subheader("Offline Signature Verification (DocSignatureNet):")
    st.warning("Note: This is a placeholder. For real-world use, you would train DocSignatureNet on a dedicated offline signature dataset and load pre-trained weights.")

    uploaded_doc_signature = st.file_uploader("Upload Document with Signature for DocSignatureNet (PNG, JPG)", type=["png", "jpg", "jpeg"], key="doc_sig")

    if uploaded_doc_signature:
        with st.spinner("Analyzing signature with DocSignatureNet..."):
            # Read uploaded file into PIL Image object
            img_doc_sig = Image.open(io.BytesIO(uploaded_doc_signature.getvalue())).convert("RGB")
            st.image(img_doc_sig, caption="Document with Signature", width='stretch')

            doc_sig_tensor = docsignature_transform(img_doc_sig).unsqueeze(0)
            
            # Forward pass
            output = docsignature_net(doc_sig_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            class_labels = {0: "Genuine", 1: "Forged"} # Assuming 0: Genuine, 1: Forged
            st.write(f"DocSignatureNet Prediction: **{class_labels.get(predicted_class, 'Unknown')}** (Confidence: {probabilities[0][predicted_class].item():.4f})")
            
            # os.remove(doc_sig_path)

    # Initialize ManTraNet
    mantra_net = ManTraNet(num_classes=2) # 1 class (tampered) + background
    mantra_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    st.subheader("Tamper Detection (ManTraNet):")
    st.warning("Note: This is a placeholder. For real-world use, you would train ManTraNet on a dedicated tamper detection dataset and load pre-trained weights.")

    uploaded_mantra_image = st.file_uploader("Upload Image for Tamper Detection (ManTraNet) (PNG, JPG)", type=["png", "jpg", "jpeg"], key="mantra_img")

    if uploaded_mantra_image:
        with st.spinner("Analyzing image for tamper using ManTraNet..."):
            # Read uploaded file into PIL Image object
            img_mantra = Image.open(io.BytesIO(uploaded_mantra_image.getvalue())).convert("RGB")
            st.image(img_mantra, caption="Original Image for ManTraNet", width='stretch')

            mantra_tensor = mantra_transform(img_mantra).unsqueeze(0)

            output = mantra_net(mantra_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            class_labels = {0: "Genuine", 1: "Tampered"} # Assuming 0: Genuine, 1: Tampered
            st.write(f"ManTraNet Prediction: **{class_labels.get(predicted_class, 'Unknown')}** (Confidence: {probabilities[0][predicted_class].item():.4f})")
            
            # os.remove(mantra_image_path)

    # Initialize EXIF_CV_Hybrid
    exif_cv_detector = EXIF_CV_Hybrid()

    st.subheader("Tamper Detection (EXIF-based + CV Hybrid):")
    st.warning("Note: This model uses EXIF metadata and basic CV cues. It is a heuristic approach and should be combined with more robust methods for critical applications.")

    uploaded_exif_cv_image = st.file_uploader("Upload Image for Tamper Detection (EXIF + CV Hybrid) (PNG, JPG)", type=["png", "jpg", "jpeg"], key="exif_cv_img")

    if uploaded_exif_cv_image:
        with st.spinner("Analyzing image for tamper using EXIF + CV Hybrid..."):
            # Read uploaded file into PIL Image object
            img_exif_cv_pil = Image.open(io.BytesIO(uploaded_exif_cv_image.getvalue())).convert("RGB")
            img_exif_cv_opencv = np.array(img_exif_cv_pil)
            img_exif_cv_opencv = cv2.cvtColor(img_exif_cv_opencv, cv2.COLOR_RGB2BGR)

            st.image(img_exif_cv_pil, caption="Original Image for EXIF + CV Analysis", width='stretch')

            is_tampered, indicators = exif_cv_detector.detect_tamper(img_exif_cv_pil, img_exif_cv_opencv)

            if is_tampered:
                st.error(f"**Tamper Detected!**")
                for ind in indicators:
                    st.write(f"- {ind}")
            else:
                st.success("No obvious tamper indicators found.")
            
            # os.remove(exif_cv_image_path)

    # Initialize VisionTransformerTamperDetector
    vit_detector = VisionTransformerTamperDetector(num_classes=2)

    st.subheader("Tamper Detection (Vision Transformer - ViT):")
    st.warning("Note: This is a placeholder. For real-world use, you would fine-tune the ViT model on a dedicated forgery/deepfake dataset.")

    uploaded_vit_image = st.file_uploader("Upload Image for Tamper Detection (Vision Transformer) (PNG, JPG)", type=["png", "jpg", "jpeg"], key="vit_img")

    if uploaded_vit_image:
        with st.spinner("Analyzing image for tamper using Vision Transformer..."):
            # Read uploaded file into PIL Image object
            img_vit = Image.open(io.BytesIO(uploaded_vit_image.getvalue())).convert("RGB")
            st.image(img_vit, caption="Original Image for ViT Analysis", width='stretch')

            # Preprocess image for ViT
            inputs = vit_detector.feature_extractor(images=img_vit, return_tensors="pt")
            pixel_values = inputs.pixel_values

            # Forward pass
            logits = vit_detector(pixel_values)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            class_labels = {0: "Genuine", 1: "Tampered"} # Assuming 0: Genuine, 1: Tampered
            st.write(f"Vision Transformer Prediction: **{class_labels.get(predicted_class, 'Unknown')}** (Confidence: {probabilities[0][predicted_class].item():.4f})")
            
            # os.remove(vit_image_path)

    # Initialize Autoencoder and Discriminator
    autoencoder_model = Autoencoder()
    discriminator_model = Discriminator()
    auto_gan_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    st.subheader("Tamper Detection (Autoencoder / GAN Discriminator):")
    st.warning("Note: This is a placeholder. For real-world use, you would train these models on a dataset of genuine images to learn normal patterns.")

    uploaded_auto_gan_image = st.file_uploader("Upload Image for Tamper Detection (Autoencoder / GAN) (PNG, JPG)", type=["png", "jpg", "jpeg"], key="auto_gan_img")

    if uploaded_auto_gan_image:
        with st.spinner("Analyzing image for tamper using Autoencoder / GAN Discriminator..."):
            # Read uploaded file into PIL Image object
            img_auto_gan = Image.open(io.BytesIO(uploaded_auto_gan_image.getvalue())).convert("RGB")
            st.image(img_auto_gan, caption="Original Image for Autoencoder / GAN Analysis", width='stretch')

            auto_gan_tensor = auto_gan_transform(img_auto_gan).unsqueeze(0)

            # Autoencoder reconstruction
            reconstructed_image = autoencoder_model(auto_gan_tensor)
            reconstruction_error = F.mse_loss(reconstructed_image, auto_gan_tensor, reduction='none')
            reconstruction_error_mean = reconstruction_error.mean().item()

            # Discriminator output
            discriminator_output = discriminator_model(auto_gan_tensor).item()

            st.write(f"Autoencoder Reconstruction Error (MSE): {reconstruction_error_mean:.4f}")
            st.write(f"GAN Discriminator Output (Probability of Genuine): {discriminator_output:.4f}")

            # Heuristic for anomaly detection (these thresholds need to be tuned)
            autoencoder_threshold = 0.05 # Example threshold for reconstruction error
            discriminator_threshold = 0.5 # Example threshold for discriminator output

            is_tampered_ae = reconstruction_error_mean > autoencoder_threshold
            is_tampered_gan = discriminator_output < discriminator_threshold

            if is_tampered_ae or is_tampered_gan:
                st.error("**Tamper Detected!** (Based on Autoencoder or GAN Discriminator)")
                if is_tampered_ae:
                    st.write(f"- Autoencoder: High reconstruction error ({reconstruction_error_mean:.4f} > {autoencoder_threshold})")
                if is_tampered_gan:
                    st.write(f"- GAN Discriminator: Low probability of genuine ({discriminator_output:.4f} < {discriminator_threshold})")
            else:
                st.success("No obvious tamper indicators found.")
            
            # os.remove(auto_gan_image_path)

    DO_EXTRACT = True
    entities_json_str = None

    if DO_EXTRACT:
        entities_json_str = run_gemini_repair(pil_image, raw_text, prompt_mode="extract")
        st.subheader("Extracted Entities:")
        st.code(entities_json_str, language='json') # Add this line to display raw JSON
        if entities_json_str:
            try:
                entities = json.loads(entities_json_str)
                st.json(entities)
                
                st.subheader("Validation Status:")
                
                # Passport Specific Validations
                if document_type == "Passport":
                    st.subheader("Passport Validation:")
                    expected_passport_fields = ["passport_number", "full_name", "date_of_birth", "place_of_birth", "date_of_issue", "date_of_expiry", "nationality", "gender", "passport_type"]
                    
                    passport_validation_data = []
                    for field in expected_passport_fields:
                        value = entities.get(field)
                        field_name = field.replace("_", " ").title()
                        status = "Missing"

                        if value:
                            if field == "passport_number":
                                is_valid = len(value) >= 7 and len(value) <= 9 # Typical passport number length
                                status = 'Valid' if is_valid else 'Invalid'
                            elif field == "date_of_expiry":
                                status = check_expiry(value, is_expiry_date=True)
                            elif "date" in field: # For date_of_birth, date_of_issue
                                status = check_expiry(value, is_expiry_date=False)
                            else:
                                status = "Valid" # Basic presence check
                        passport_validation_data.append([field_name, value if value else "N/A", status])
                    
                    st.table(passport_validation_data)
                
                # Aadhaar Specific Validations
                elif document_type == "Aadhaar":
                    st.subheader("Aadhaar Validation:")
                    expected_aadhaar_fields = ["aadhaar_number", "full_name", "date_of_birth", "gender", "address"]

                    aadhaar_validation_data = []
                    for field in expected_aadhaar_fields:
                        value = entities.get(field)
                        field_name = field.replace("_", " ").title()
                        status = "Missing"

                        if value:
                            if field == "aadhaar_number":
                                is_valid = len(value) == 12 and value.isdigit() # Aadhaar numbers are 12 digits
                                status = 'Valid' if is_valid else 'Invalid'
                            elif "date" in field: # For date_of_birth
                                status = check_expiry(value, is_expiry_date=False)
                            else:
                                status = "Valid" # Basic presence check
                        aadhaar_validation_data.append([field_name, value if value else "N/A", status])
                    
                    st.table(aadhaar_validation_data)

                # Placeholder for advanced visual checks
                st.subheader("Advanced Visual Checks (Future Development):")
                st.info("\n- **Stamp/Seal/Signature Verification**: Requires specialized computer vision models for authentication.\n- **Tamper Detection**: Requires advanced image forensics techniques to identify manipulations.\n")

                st.subheader("Experimental Visual Analysis (Gemini-based):")
                st.warning("\nThis is an experimental, AI-driven visual analysis based on Gemini's interpretation of the image. It is NOT a robust or secure solution for forgery detection or official verification of stamps/signatures. For critical applications, specialized computer vision models and image forensics techniques are required.\n")
                
                visual_analysis_results = run_gemini_visual_analysis(pil_image, raw_text)
                st.write(visual_analysis_results)

            except json.JSONDecodeError:
                st.error("Error: Could not decode entities JSON.")
        else:
            st.write("No entities extracted.")
    
    # Clean up the temporary file
    # os.remove(temp_image_path)

st.markdown("""
---
### How to Run This Application:

1.  **Ensure you have Tesseract OCR installed** and configured. If on Windows, update `TESSERACT_CMD` in `app.py`.
2.  **Install Python dependencies**: Navigate to the `Nyaya_Samiti/model` directory and run:
    ```bash
    pip install -r requirements.txt
    ```
    Then, navigate back to the `Nyaya_Samiti` directory.
3.  **Set your Google Gemini API Key**: Create a `.env` file in the `Nyaya_Samiti` directory with your API key:
    ```
    GEMINI_API_KEY="YOUR_API_KEY"
    ```
4.  **Run the Streamlit app**: From the `Nyaya_Samiti` directory, execute:
    ```bash
    streamlit run app.py
    ```

**Note**: The models integrated (Siamese, DocSignatureNet, ManTraNet, ViT, Autoencoder/GAN) are placeholders and require proper training on relevant datasets for real-world performance. They are demonstrated here for conceptual integration within the Streamlit application.
""")
