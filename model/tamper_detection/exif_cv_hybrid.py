
import os
from PIL import Image
import piexif
import numpy as np
import cv2
import io

class EXIF_CV_Hybrid:
    def __init__(self):
        pass

    def analyze_exif(self, image_bytes):
        try:
            exif_dict = piexif.load(image_bytes)
            print(f"EXIF data:")
            for ifd_name in exif_dict:
                print(f"{ifd_name}:")
                for key, value in exif_dict[ifd_name].items():
                    try:
                        print(f"  {piexif.TAGS[ifd_name][key]['name']}: {value}")
                    except KeyError:
                        print(f"  {key}: {value}")
            return exif_dict
        except piexif.InvalidImageDataError:
            print(f"No EXIF data found or invalid EXIF data in image.")
            return None
        except Exception as e:
            print(f"Error reading EXIF data from image: {e}")
            return None

    def analyze_cv_cues(self, img_np):
        try:
            img = img_np
            if img is None:
                raise ValueError(f"Input image is None.")
            
            height, width, channels = img.shape
            # For in-memory image, file_size is not directly applicable, or would need to be passed separately if relevant
            file_size = 0 # Placeholder, as file_size is not directly available from numpy array

            print(f"CV Cues:")
            print(f"  Dimensions: {width}x{height}")
            print(f"  File Size: {file_size} bytes")

            # Example of a simple CV cue: check for uniform color patches (might indicate tampering)
            # This is a very basic example and would need more sophisticated algorithms.
            mean_color = np.mean(img, axis=(0, 1))
            std_color = np.std(img, axis=(0, 1))
            print(f"  Mean Color: {mean_color}")
            print(f"  Standard Deviation of Color: {std_color}")

            return {
                "dimensions": (width, height),
                "file_size": file_size,
                "mean_color": mean_color.tolist(),
                "std_color": std_color.tolist()
            }
        except Exception as e:
            print(f"Error analyzing CV cues: {e}")
            return None

    def detect_tamper(self, pil_image, opencv_image):
        print(f"\n--- Detecting Tamper ---")

        # Get image bytes for EXIF analysis
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="jpeg")
        img_bytes = img_bytes.getvalue()

        exif_data = self.analyze_exif(img_bytes)
        cv_cues = self.analyze_cv_cues(opencv_image)

        tamper_indicators = []

        if exif_data:
            # Example: Check for software inconsistencies in EXIF
            if piexif.ImageIFD.Software in exif_data["0th"]:
                software = exif_data["0th"][piexif.ImageIFD.Software].decode('utf-8')
                if "photoshop" in software.lower():
                    tamper_indicators.append("EXIF: Image processed by Photoshop (potential manipulation).")
            # More EXIF checks can be added here (e.g., inconsistencies in dates, original dimensions)

        if cv_cues:
            # Example: Simple check for very low color std deviation (might indicate artificially created regions)
            # This is highly heuristic and needs fine-tuning with real data.
            if np.mean(cv_cues["std_color"]) < 5.0:
                tamper_indicators.append("CV: Low color standard deviation (potential uniform patch manipulation).")

        if tamper_indicators:
            print(f"Tamper detected in the image based on the following indicators:")
            for indicator in tamper_indicators:
                print(f"- {indicator}")
            return True, tamper_indicators
        else:
            print(f"No obvious tamper indicators found in the image.")
            return False, []

if __name__ == '__main__':
    # Example Usage and Setup Notes
    # 1. Install necessary libraries: pip install piexif opencv-python numpy pillow
    # 2. Prepare sample images, some genuine, some with known tampering (e.g., Photoshop edited).

    detector = EXIF_CV_Hybrid()

    genuine_image = Image.new('RGB', (800, 600), color = 'white')
    tampered_img = Image.new('RGB', (800, 600), color = 'lightgray')

    # Add some dummy EXIF data to tampered image (optional, for demonstration of EXIF analysis)
    # To add EXIF to in-memory PIL Image, we need to save it to bytes first, add EXIF, then load back
    # For simplicity, we'll keep the EXIF example as-is with a temporary file for now.
    temp_tampered_path = "temp_tampered_with_exif.jpg"
    tampered_img.save(temp_tampered_path)

    zeroth_ifd = {
        piexif.ImageIFD.Make: b"Dummy Camera",
        piexif.ImageIFD.Software: b"Adobe Photoshop CC 2023"
    }
    exif_bytes = piexif.dump({"0th": zeroth_ifd})
    piexif.insert(exif_bytes, temp_tampered_path)

    tampered_img_with_exif = Image.open(temp_tampered_path).convert("RGB")
    os.remove(temp_tampered_path)

    genuine_opencv = np.array(genuine_image)
    genuine_opencv = cv2.cvtColor(genuine_opencv, cv2.COLOR_RGB2BGR)

    tampered_opencv = np.array(tampered_img_with_exif)
    tampered_opencv = cv2.cvtColor(tampered_opencv, cv2.COLOR_RGB2BGR)

    # Test on genuine image
    is_tampered_genuine, indicators_genuine = detector.detect_tamper(genuine_image, genuine_opencv)
    print(f"Result for genuine image: Tampered={is_tampered_genuine}, Indicators={indicators_genuine}")

    # Test on tampered image
    is_tampered_tampered, indicators_tampered = detector.detect_tamper(tampered_img_with_exif, tampered_opencv)
    print(f"Result for tampered image: Tampered={is_tampered_tampered}, Indicators={indicators_tampered}")

    # Clean up dummy images
    # os.remove(genuine_image_path)
    # os.remove(tampered_image_path)
    print("EXIF-based + CV Hybrid model outlined successfully.")
