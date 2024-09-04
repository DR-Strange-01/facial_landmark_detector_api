from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import dlib
import numpy as np
import requests
from io import BytesIO

app = FastAPI()

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

class ImageURL(BaseModel):
    url: str

def is_image_blurry(image, threshold=100):
    """Checking if the image is blurry using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_image_too_bright_or_dark(image, bright_threshold=220, dark_threshold=30):
    """Checking if the image is too bright or too dark based on average pixel intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > bright_threshold or mean_brightness < dark_threshold

@app.post("/process_image/")
async def process_image(image_url: ImageURL):
    try:
        # Get the image from the URL
        response = requests.get(image_url.url)
        image = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image URL or format.")

        # Check image quality
        if is_image_blurry(image) or is_image_too_bright_or_dark(image):
            return {"message": "POOR LIGHTNING CONDITIONS! Kindly upload a clear image or use clear daylight for a selfie."}

        # Convert to grayscale for facial detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)


        for face in faces:
            landmarks = predictor(gray, face)
            for i in range(68):
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(image, (x, y), 3, (255, 255, 255), -1)

        # Save processed image
        output_path = 'Result.jpg'
        cv2.imwrite(output_path, image)

        return {"message": "Image processed successfully", "Result": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
