from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mb7m4Bcuxtbp5gAuPR0j"
)

car_image_path = 'test-images/car.png'
car_image = cv2.imread(car_image_path)

if car_image is None:
    print(f"Error: Could not load image {car_image_path}")
    exit()

response = CLIENT.infer(car_image_path, model_id="vehicle-registration-plates-trudk/2")
predictions = response["predictions"]

license_plate_bbox = None
for pred in predictions:
    if pred["class"] == "License_Plate":
        x, y, width, height = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1, x2, y2 = x - width // 2, y - height // 2, x + width // 2, y + height // 2
        license_plate_bbox = (x1, y1, x2, y2)
        break

if license_plate_bbox is None:
    print("License plate not found.")
    exit()

x1, y1, x2, y2 = license_plate_bbox
car_image_with_white_plate = car_image.copy()
cv2.rectangle(car_image_with_white_plate, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

logo_image_path = 'logo/Logo.png'
logo_image = cv2.imread(logo_image_path, cv2.IMREAD_UNCHANGED)

if logo_image is None:
    print(f"Error: Could not load logo {logo_image_path}")
    exit()

logo_resized = cv2.resize(
    logo_image,
    (x2 - x1, y2 - y1)
)

if logo_resized.shape[2] == 4:
    alpha_logo = logo_resized[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_logo

    for c in range(0, 3):
        car_image_with_white_plate[y1:y2, x1:x2, c] = (
            alpha_logo * logo_resized[:, :, c] +
            alpha_background * car_image_with_white_plate[y1:y2, x1:x2, c]
        )
else:
    car_image_with_white_plate[y1:y2, x1:x2] = logo_resized

output_image_path = 'output/car_with_logo.png'
cv2.imwrite(output_image_path, car_image_with_white_plate)
cv2.imshow('Car with Logo', car_image_with_white_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()
