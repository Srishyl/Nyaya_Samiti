import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
import pytesseract
import google.generativeai as genai
from dotenv import load_dotenv
import io
import json
from datetime import datetime
import re
import torch
from torchvision import transforms
import torch.nn.functional as F

# Model Imports
from model.signature_verification.siamese_network import SiameseNetwork
from model.object_detection.faster_rcnn import StampSealDetector
from model.object_detection.embedding_matcher import EmbeddingMatcher
from model.signature_verification.docsignaturenet import DocSignatureNet
from model.tamper_detection.exif_cv_hybrid import EXIF_CV_Hybrid
from model.tamper_detection.noise_inconsistency import ela_image, ELA_CNN
from model.tamper_detection.mantra_net import ManTraNet
from model.tamper_detection.vision_transformers import VisionTransformerTamperDetector
from model.tamper_detection.autoencoder_gan import Autoencoder, Discriminator
from model.document_classifier import DocumentClassifier

# --- Configuration ---
# Tesseract OCR path (update if necessary)
TESSERACT_CMD = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # For Windows
# TESSERACT_CMD = '/usr/local/bin/tesseract' # For macOS/Linux

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
OCR_LANGS = "eng+hin" # Default OCR languages

# --- Gemini API Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Prompt Templates for Document Validation ---
PASSPORT_PROMPT_TEMPLATE = """
As a legal document validator, your task is to meticulously examine the provided Indian Passport document image and its extracted text. Adhere strictly to the following format and validation rules for an Indian Passport:

Expected Indian Passport Fields (Front Page):
- Passport Number (typically 7-9 alphanumeric characters)
- Type (always 'P' for Passport)
- Code (always 'IND' for India)
- Surname
- Given Name(s)
- Nationality (always 'Indian')
- Date of Birth (DD/MM/YYYY format)
- Place of Birth
- Sex (M/F)
- Date of Issue (DD/MM/YYYY format)
- Date of Expiry (DD/MM/YYYY format - must be after date of issue and in the future for validity)
- Place of Issue
- File Number

Additional Validity Checks:
- The image should appear genuine, with no signs of tampering, blurriness, or obvious digital manipulation.
- The text extracted by OCR should be consistent with the visual elements of the passport.
- Dates (Date of Birth, Issue, Expiry) must be in a logical sequence.
- The passport must not be expired.

Based on your analysis, state clearly whether the document is 'Valid' or 'Invalid'. Provide a detailed explanation for your decision, specifically highlighting any discrepancies, missing information, signs of tampering, or formatting issues. If it's valid, explain why. If it's invalid, list all reasons. Focus on the formal validity and integrity of the document, not on the identity of the person.

Here is the extracted text from the document:
{extracted_text}

Here are the results from the automated computer vision checks for tampering:
{tamper_detection_results}

Here are the results from the automated computer vision checks for object detection (stamps/seals):
{object_detection_results}

Here are the results from the rule-based format validation:
{format_validation_results}

Your verdict (Valid/Invalid) and detailed explanation:
"""

AADHAAR_PROMPT_TEMPLATE = """
As a legal document validator, your task is to meticulously examine the provided Indian Aadhaar Card document image and its extracted text. Adhere strictly to the following format and validation rules for an Indian Aadhaar Card:

Expected Indian Aadhaar Card Fields (Front Side):
- Aadhaar Number (12 digits, typically displayed in blocks of four: XXXX XXXX XXXX)
- Name (Full Name)
- Year of Birth or Date of Birth (YYYY or DD-MM-YYYY format)
- Gender (Male/Female/Transgender)

Expected Indian Aadhaar Card Fields (Back Side - if applicable/available):
- Address (Full Residential Address)
- Guardian/Parent Name (if applicable)

Additional Validity Checks:
- The image should appear genuine, with no signs of tampering, blurriness, or obvious digital manipulation.
- The text extracted by OCR should be consistent with the visual elements of the Aadhaar card.
- The Aadhaar number must be exactly 12 digits and numeric.
- The card should display the official UIDAI logo.

Based on your analysis, state clearly whether the document is 'Valid' or 'Invalid'. Provide a detailed explanation for your decision, specifically highlighting any discrepancies, missing information, signs of tampering, or formatting issues. If it's valid, explain why. If it's invalid, list all reasons. Focus on the formal validity and integrity of the document, not on the identity of the person.

Here is the extracted text from the document:
{extracted_text}

Here are the results from the automated computer vision checks for tampering:
{tamper_detection_results}

Here are the results from the automated computer vision checks for object detection (stamps/seals):
{object_detection_results}

Here are the results from the rule-based format validation:
{format_validation_results}

Your verdict (Valid/Invalid) and detailed explanation:
"""

# --- Utility Functions ---

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

def check_expiry(date_string, is_expiry_date=True):
    fmts = ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d']
    parsed_date = None
    for fmt in fmts:
        try:
            parsed_date = datetime.strptime(date_string, fmt).date()
            break
        except ValueError:
            continue

    if parsed_date is None:
        return "Invalid Date Format"

    if is_expiry_date:
        if parsed_date < datetime.now().date():
            return "Expired"
        else:
            return "Valid"
    else:
        # For date of birth, just check if it's a valid date and not in the future
        if parsed_date > datetime.now().date():
            return "Future Date (Invalid)"
        else:
            return "Valid"

def run_gemini_repair(image_pil, raw_text, prompt_mode="clean"):
    """
    prompt_mode:
      - "clean": produce clean reconstructed text using both inputs.
      - "extract": return structured fields (entities) as JSON.
    """
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction="You are an expert OCR post-processor. Your task is to accurately reconstruct text from document images and extract structured entities.")

    if prompt_mode == "clean":
        prompt_parts = [
            image_pil,
            f"Given the image and the raw OCR text below, reconstruct the most accurate text. "
            f"Preserve reading order, headers, tables (as Markdown), and line breaks. "
            f"If handwriting is present, read from the image to fill missing/incorrect words. "
            f"Do not invent content that is not visible in the image.\\n\\nRaw OCR Text:\\n{raw_text}"
        ]
    elif prompt_mode == "extract":
        prompt_parts = [
            image_pil,
            f"Extract key fields from the document as JSON. For Aadhaar cards, extract 'aadhaar_number', 'full_name', 'date_of_birth', 'gender', 'address'. "
            f"For Passports, extract 'passport_number', 'full_name', 'date_of_birth', 'place_of_birth', 'date_of_issue', 'date_of_expiry', 'nationality', 'gender', 'passport_type'. "
            f"For other documents, use general keys like 'names', 'ids', 'dates', 'addresses', 'emails', 'phones', 'stamps_or_seals'. "
            f"When unknown, use null or empty arrays. Do not include any commentary. Only return the JSON object.\\n\\nRaw OCR Text:\\n{raw_text}"
        ]
    else:
        return "Invalid prompt_mode specified."

    response = model.generate_content(prompt_parts)
    full_response_text = response.text

    if prompt_mode == "extract":
        try:
            # Attempt to parse as JSON. Gemini sometimes adds ```json wrappers.
            if full_response_text.strip().startswith('```json') and full_response_text.strip().endswith('```'):
                json_str = full_response_text.strip()[7:-3].strip()
            else:
                json_str = full_response_text.strip()
            return json.loads(json_str)
        except json.JSONDecodeError:
            st.error(f"Failed to parse Gemini's entity extraction as JSON: {full_response_text}")
            return {}
    return full_response_text

# --- Validation Functions ---
def validate_passport_fields(entities):
    validation_results = {}

    # Passport Number: 7-9 alphanumeric characters
    passport_number = entities.get('passport_number', '')
    if passport_number:
        if 7 <= len(passport_number) <= 9 and passport_number.isalnum():
            validation_results['Passport Number Format'] = 'Valid'
        else:
            validation_results['Passport Number Format'] = 'Invalid (expected 7-9 alphanumeric)'
    else:
        validation_results['Passport Number'] = 'Missing'

    # Nationality: Must be 'Indian'
    nationality = entities.get('nationality', '')
    if nationality:
        if nationality.lower() == 'indian':
            validation_results['Nationality'] = 'Valid'
        else:
            validation_results['Nationality'] = f'Invalid (expected "Indian", got "{nationality}")'
    else:
        validation_results['Nationality'] = 'Missing'

    # Dates: DOB, Issue, Expiry (DD/MM/YYYY format, logical order, not expired)
    dob_str = entities.get('date_of_birth', '')
    issue_date_str = entities.get('date_of_issue', '')
    expiry_date_str = entities.get('date_of_expiry', '')

    def parse_date(date_string):
        try:
            return datetime.strptime(date_string, '%d/%m/%Y')
        except ValueError:
            return None

    dob = parse_date(dob_str)
    issue_date = parse_date(issue_date_str)
    expiry_date = parse_date(expiry_date_str)

    if dob:
        validation_results['Date of Birth Format'] = 'Valid'
    else:
        validation_results['Date of Birth'] = 'Invalid (expected DD/MM/YYYY)'

    if issue_date:
        validation_results['Date of Issue Format'] = 'Valid'
    else:
        validation_results['Date of Issue'] = 'Invalid (expected DD/MM/YYYY)'

    if expiry_date:
        validation_results['Date of Expiry Format'] = 'Valid'
    else:
        validation_results['Date of Expiry'] = 'Invalid (expected DD/MM/YYYY)'

    if issue_date and expiry_date:
        if issue_date < expiry_date:
            validation_results['Issue Before Expiry'] = 'Valid'
        else:
            validation_results['Issue Before Expiry'] = 'Invalid (issue date must be before expiry date)'
        
        if expiry_date > datetime.now():
            validation_results['Not Expired'] = 'Valid'
        else:
            validation_results['Not Expired'] = 'Invalid (document has expired)'
    else:
        validation_results['Expiry Check'] = 'Cannot perform (missing date info)'

    # Gender: M/F
    gender = entities.get('gender', '')
    if gender:
        if gender.upper() in ['M', 'F']:
            validation_results['Gender Format'] = 'Valid'
        else:
            validation_results['Gender Format'] = 'Invalid (expected M or F)'
    else:
        validation_results['Gender'] = 'Missing'
    
    return validation_results

def validate_aadhaar_fields(entities):
    validation_results = {}

    # Aadhaar Number: 12 digits, numeric
    aadhaar_number = entities.get('aadhaar_number', '')
    if aadhaar_number:
        # Remove spaces if present for validation
        clean_aadhaar = aadhaar_number.replace(' ', '')
        if len(clean_aadhaar) == 12 and clean_aadhaar.isdigit():
            validation_results['Aadhaar Number Format'] = 'Valid'
        else:
            validation_results['Aadhaar Number Format'] = 'Invalid (expected 12 numeric digits)'
    else:
        validation_results['Aadhaar Number'] = 'Missing'

    # Name: Presence check
    full_name = entities.get('full_name', '')
    if full_name and len(full_name) > 3: # Basic length check
        validation_results['Full Name'] = 'Valid'
    else:
        validation_results['Full Name'] = 'Missing or Too Short'

    # Date of Birth/Year of Birth: Presence and basic format check (YYYY or DD-MM-YYYY)
    dob_yob = entities.get('date_of_birth', entities.get('year_of_birth', ''))
    if dob_yob:
        # Simple regex for YYYY or DD-MM-YYYY. More robust parsing can be added.
        if re.match(r'^\d{4}$|^(0[1-9]|[12][0-9]|3[01])[-/.](0[1-9]|1[012])[-/.](19|20)\d{2}$', dob_yob):
            validation_results['Date/Year of Birth Format'] = 'Valid'
        else:
            validation_results['Date/Year of Birth Format'] = 'Invalid (expected YYYY or DD-MM-YYYY)'
    else:
        validation_results['Date/Year of Birth'] = 'Missing'
    
    # Gender: Male/Female/Transgender
    gender = entities.get('gender', '')
    if gender:
        if gender.lower() in ['male', 'female', 'transgender', 'm', 'f', 't']:
            validation_results['Gender Format'] = 'Valid'
        else:
            validation_results['Gender Format'] = 'Invalid (expected Male/Female/Transgender)'
    else:
        validation_results['Gender'] = 'Missing'

    # Address: Presence check (assuming it's extracted for back side)
    address = entities.get('address', '')
    if address and len(address) > 10: # Basic length check for a meaningful address
        validation_results['Address'] = 'Valid'
    else:
        validation_results['Address'] = 'Missing or Too Short'

    return validation_results


# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="Legal Document Validation", page_icon="⚖️")

# Initialize session state for predicted_doc_type if not already set
if 'predicted_doc_type' not in st.session_state:
    st.session_state.predicted_doc_type = "Unknown"

# --- Streamlit UI ---
st.title("Document OCR and Validation")

# Initialize models (These should ideally be loaded once or cached)
# Stamp Seal Detector
stamp_seal_detector = StampSealDetector()
# Embedding Matcher
embedding_matcher = EmbeddingMatcher(model_name='resnet18') # Can be 'efficientnet_b0'
# DocSignatureNet
docsignature_net = DocSignatureNet(num_classes=2)
# ManTra-Net
mantra_net = ManTraNet(num_classes=2)
# EXIF-CV Hybrid
exif_cv_hybrid_detector = EXIF_CV_Hybrid()
# ELA + CNN for Noise Inconsistency
ela_cnn_detector = ELA_CNN(num_classes=2)
# Vision Transformer Tamper Detector
vit_tamper_detector = VisionTransformerTamperDetector(num_classes=2)
# Autoencoder and GAN Discriminator
autoencoder = Autoencoder()
discriminator = Discriminator()
# Document Classifier
document_classifier = DocumentClassifier(num_classes=2) # 0 for Aadhaar, 1 for Passport
class_names = {0: "Aadhaar Card", 1: "Indian Passport"}

# Placeholder for loading trained model weights:
# try:
#     document_classifier.load_state_dict(torch.load("path/to/document_classifier_weights.pth"))
#     document_classifier.eval()
# except FileNotFoundError:
#     st.sidebar.warning("Document classifier weights not found. Using untrained model.")


uploaded_file = st.file_uploader("Upload Document Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Process uploaded image
    img_bytes = uploaded_file.getvalue()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    opencv_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Document Classification
    with st.spinner("Classifying document type..."):
        predicted_class_idx, probabilities = document_classifier.predict(pil_image)
        predicted_doc_type = class_names.get(predicted_class_idx, "Unknown")
        st.success(f"Document Classified as: **{predicted_doc_type}** (Confidence: {probabilities[predicted_class_idx]:.2f})")
        st.session_state.predicted_doc_type = predicted_doc_type

    st.image(pil_image, caption="Uploaded Image.", width='stretch')

    # --- OCR and Gemini Text Repair ---
    st.subheader("1. OCR & Text Processing")
    col1_ocr, col2_ocr = st.columns(2)

    with col1_ocr:
        st.info("Performing OCR...")
        pre = preprocess_for_ocr(opencv_image)
        raw_text = tesseract_ocr(pre, psm=6, oem=3)
        st.subheader("Raw OCR Text:")
        st.write(raw_text[:500] + "...") # Display first 500 chars

    with col2_ocr:
        st.info("Running Gemini Text Repair...")
        # Get cleaned text (basic reconstruction)
        gemini_cleaned_text = run_gemini_repair(pil_image, raw_text, prompt_mode="clean")
        st.subheader("Gemini Cleaned Text:")
        st.write(gemini_cleaned_text[:500] + "...") # Display first 500 chars

    # --- Entity Extraction and Rule-Based Validation ---
    st.subheader("2. Entity Extraction & Rule-Based Validation")
    with st.spinner("Extracting entities and performing rule-based validation..."):
        extracted_entities = run_gemini_repair(pil_image, raw_text, prompt_mode="extract")
        
        st.subheader("Extracted Entities (Gemini):")
        st.json(extracted_entities)

        format_validation_results = {}
        if predicted_doc_type == "Indian Passport":
            format_validation_results = validate_passport_fields(extracted_entities)
            st.subheader("Passport Rule-Based Validation:")
            st.table(format_validation_results.items())
        elif predicted_doc_type == "Aadhaar Card":
            format_validation_results = validate_aadhaar_fields(extracted_entities)
            st.subheader("Aadhaar Rule-Based Validation:")
            st.table(format_validation_results.items())
        else:
            st.info(f"No specific rule-based validation for detected document type: {predicted_doc_type}.")
    
    # Format validation results for Gemini prompt
    formatted_validation_results = ""
    for k, v in format_validation_results.items():
        formatted_validation_results += f"- {k}: {v}\\n"
    if not formatted_validation_results:
        formatted_validation_results = "No specific rule-based validation performed or no results found."


    # --- Object Detection (Stamps/Seals) ---
    st.subheader("3. Stamp/Seal Detection")
    with st.spinner("Detecting stamps/seals..."):
        detections = stamp_seal_detector.detect(pil_image)
        if detections:
            visualized_img = stamp_seal_detector.visualize_detections(pil_image.copy(), detections)
            st.image(visualized_img, caption="Detected Stamps/Seals", width='stretch')
            st.success(f"Detected {len(detections)} stamp/seal regions.")
            object_detection_results = "Detected stamp/seal bounding boxes and scores:\\n"
            for i, det in enumerate(detections):
                object_detection_results += f"- Box: {det['box']}, Score: {det['score']:.2f}, Label: {det['label']}\\n"
        else:
            st.info("No stamps or seals detected.")
            object_detection_results = "No stamps or seals detected."

    # --- Tamper Detection ---
    st.subheader("4. Tamper Detection")
    tamper_detection_results = ""

    # ManTra-Net (Conceptual)
    with st.spinner("Running ManTra-Net (conceptual) for tamper detection..."):
        # MantraNet input expects a specific tensor format, for now, conceptual output
        # mantr-input = transforms.ToTensor()(pil_image).unsqueeze(0)
        # mantra_output = mantra_net(mantr-input)
        # For conceptual display:
        st.info("ManTra-Net is a conceptual model here. Real implementation would involve training and inference.")
        mantra_prediction = "Conceptual: No overt tampering detected by ManTra-Net."
        tamper_detection_results += f"ManTra-Net: {mantra_prediction}\\n"
    
    # EXIF-CV Hybrid
    with st.spinner("Running EXIF-CV Hybrid for tamper detection..."):
        exif_cv_result = exif_cv_hybrid_detector.detect_tamper(pil_image, opencv_image)
        st.json(exif_cv_result)
        tamper_detection_results += f"EXIF-CV Hybrid: {json.dumps(exif_cv_result)}\\n"

    # Vision Transformer Tamper Detector (Conceptual)
    with st.spinner("Running Vision Transformer (conceptual) for tamper detection..."):
        # vit_pixel_values = vit_tamper_detector.preprocess_image(pil_image)
        # vit_logits = vit_tamper_detector(vit_pixel_values)
        # vit_prediction = torch.argmax(vit_logits, dim=1).item()
        st.info("Vision Transformer is a conceptual model here. Real implementation would involve training and inference.")
        vit_prediction_text = "Conceptual: No overt tampering detected by Vision Transformer."
        tamper_detection_results += f"Vision Transformer: {vit_prediction_text}\\n"
    
    # Autoencoder / GAN Discriminator (Conceptual)
    with st.spinner("Running Autoencoder/GAN Discriminator (conceptual) for anomaly detection..."):
        # ae_input = transforms.ToTensor()(pil_image).unsqueeze(0)
        # ae_reconstruction = autoencoder(ae_input)
        # disc_output = discriminator(ae_input)
        st.info("Autoencoder/GAN Discriminator is a conceptual model here. Real implementation would involve training and inference.")
        ae_gan_prediction = "Conceptual: No significant anomalies detected by Autoencoder/GAN Discriminator."
        tamper_detection_results += f"Autoencoder/GAN Discriminator: {ae_gan_prediction}\n"

    # ELA + CNN for Noise Inconsistency
    with st.spinner("Running ELA + CNN for tamper detection..."):
        ela_output_img = ela_image(pil_image) # Use el-image directly with PIL Image
        # Prepare ELA image for CNN
        transform_ela = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ela_tensor = transform_ela(ela_output_img).unsqueeze(0)
        
        # Move tensor to CPU if model isn't on GPU
        ela_tensor = ela_tensor.to("cpu") 

        ela_logits = ela_cnn_detector(ela_tensor)
        ela_probabilities = F.softmax(ela_logits, dim=1)
        _, ela_predicted_class = torch.max(ela_probabilities, 1)
        
        ela_prediction_text = "Genuine" if ela_predicted_class.item() == 0 else "Tampered"
        st.write(f"ELA + CNN Prediction: **{ela_prediction_text}** (Confidence: {ela_probabilities.squeeze()[ela_predicted_class.item()]:.2f})")
        tamper_detection_results += f"ELA + CNN: {ela_prediction_text} (Confidence: {ela_probabilities.squeeze()[ela_predicted_class.item()]:.2f})\n"

    # --- Final Gemini Document Validation ---
    st.subheader("5. Final Document Validation (Gemini)")
    with st.spinner("Asking Gemini for final document validation and explanation..."):
        st.write(f"DEBUG: Predicted Document Type from Session State: {st.session_state.predicted_doc_type}") # Debugging line
        if st.session_state.predicted_doc_type == "Indian Passport":
            prompt_template = PASSPORT_PROMPT_TEMPLATE
        elif st.session_state.predicted_doc_type == "Aadhaar Card":
            prompt_template = AADHAAR_PROMPT_TEMPLATE
        else:
            prompt_template = """
            As a legal document validator, examine the provided document image and its extracted text.
            Based on the OCR, visual checks, and any other available information, determine if the document appears valid or invalid.
            Provide a detailed explanation for your decision, highlighting any issues found.

            Here is the extracted text from the document:
            {extracted_text}

            Here are the results from the automated computer vision checks for tampering:
            {tamper_detection_results}

            Here are the results from the automated computer vision checks for object detection (stamps/seals):
            {object_detection_results}
            
            Here are the results from the rule-based format validation:
            {format_validation_results}

            Your verdict (Valid/Invalid) and detailed explanation:
            """
        
        final_gemini_prompt = prompt_template.format(
            extracted_text=gemini_cleaned_text,
            tamper_detection_results=tamper_detection_results,
            object_detection_results=object_detection_results,
            format_validation_results=formatted_validation_results
        )

        final_validation_model = genai.GenerativeModel("gemini-2.5-flash")
        final_gemini_response = final_validation_model.generate_content([pil_image, final_gemini_prompt])
        
        st.markdown("### Gemini's Final Verdict and Explanation:")
        st.write(final_gemini_response.text)

    # --- Other Features (Existing, moved to the end or commented out for new flow) ---
    # These sections would typically be integrated into the above flow or conditional on document type.

    # Signature Verification (Siamese Network):
    # This remains a separate flow due to requiring two input images.
    st.sidebar.header("Signature Verification (Siamese Network):")
    st.sidebar.warning("This is a separate flow for signature comparison. Upload two signatures here.")
    siamese_net = SiameseNetwork()
    siamese_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)), # Assuming 100x100 input for SiameseNetwork
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    uploaded_reference_signature = st.sidebar.file_uploader("Upload Reference Signature Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="ref_sig")
    uploaded_new_signature = st.sidebar.file_uploader("Upload New Signature Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="new_sig")

    if uploaded_reference_signature and uploaded_new_signature:
        with st.spinner("Verifying signatures..."):
            img_ref = Image.open(io.BytesIO(uploaded_reference_signature.getvalue())).convert("RGB")
            img_new = Image.open(io.BytesIO(uploaded_new_signature.getvalue())).convert("RGB")

            st.sidebar.image([img_ref, img_new], caption=["Reference Signature", "New Signature"], width=100)

            img_ref_tensor = siamese_transform(img_ref).unsqueeze(0)
            img_new_tensor = siamese_transform(img_new).unsqueeze(0)

            output1, output2 = siamese_net(img_ref_tensor, img_new_tensor)
            euclidean_distance = F.pairwise_distance(output1, output2)

            similarity_threshold = 1.0 # This threshold needs to be determined by training
            if euclidean_distance.item() < similarity_threshold:
                st.sidebar.success(f"Signatures are LIKELY GENUINE! Distance: {euclidean_distance.item():.4f}")
            else:
                st.sidebar.error(f"Signatures are LIKELY FORGED! Distance: {euclidean_distance.item():.4f}")

else:
    st.info("Please upload a document image to start the validation process.")
