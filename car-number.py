from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

def add_logo_to_license_plates(car_image_path, logo_image_path, output_image_path, model_id="vehicle-registration-plates-trudk/2", padding=5, border_thickness=3):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="mb7m4Bcuxtbp5gAuPR0j"
    )

    car_image = cv2.imread(car_image_path)
    if car_image is None:
        print(f"Error: Could not load image {car_image_path}")
        return

    response = CLIENT.infer(car_image_path, model_id=model_id)
    predictions = response.get("predictions", [])
    if not predictions:
        print("No license plates detected in the image.")
        return

    logo_image = cv2.imread(logo_image_path, cv2.IMREAD_UNCHANGED)
    if logo_image is None:
        print(f"Error: Could not load logo {logo_image_path}")
        return

    if car_image.shape[0] < 100 or car_image.shape[1] < 100:
        print("Warning: The image size is too small to add logos in a visually consistent way.")
        return

    for pred in predictions:
        if pred["class"] == "License_Plate":
            x, y, width, height = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            x1, y1, x2, y2 = x - width // 2, y - height // 2, x + width // 2, y + height // 2
            x1_padded, y1_padded = max(0, x1 - padding), max(0, y1 - padding)
            x2_padded, y2_padded = min(car_image.shape[1], x2 + padding), min(car_image.shape[0], y2 + padding)

            car_image[y1_padded:y2_padded, x1_padded:x2_padded] = (255, 255, 255)

            logo_height, logo_width = logo_image.shape[:2]
            scale_factor = min((x2_padded - x1_padded - 2 * border_thickness) / logo_width,
                               (y2_padded - y1_padded - 2 * border_thickness) / logo_height)
            new_logo_width, new_logo_height = int(logo_width * scale_factor), int(logo_height * scale_factor)

            logo_resized = cv2.resize(logo_image, (new_logo_width, new_logo_height))

            bordered_logo = cv2.copyMakeBorder(
                logo_resized,
                top=border_thickness,
                bottom=border_thickness,
                left=border_thickness,
                right=border_thickness,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            x_offset = x1_padded + (x2_padded - x1_padded - bordered_logo.shape[1]) // 2
            y_offset = y1_padded + (y2_padded - y1_padded - bordered_logo.shape[0]) // 2

            if bordered_logo.shape[2] == 4:
                alpha_logo = bordered_logo[:, :, 3] / 255.0
                alpha_car = 1.0 - alpha_logo

                for c in range(3):
                    car_image[y_offset:y_offset + bordered_logo.shape[0], x_offset:x_offset + bordered_logo.shape[1], c] = (
                        alpha_logo * bordered_logo[:, :, c] +
                        alpha_car * car_image[y_offset:y_offset + bordered_logo.shape[0], x_offset:x_offset + bordered_logo.shape[1], c]
                    )
            else:
                car_image[y_offset:y_offset + bordered_logo.shape[0], x_offset:x_offset + bordered_logo.shape[1]] = bordered_logo

    cv2.imwrite(output_image_path, car_image)
    print(f"Output image saved at {output_image_path}")

add_logo_to_license_plates(
    car_image_path='test-images/car_2.png',
    logo_image_path='logo/Logo.png',
    output_image_path='output/cars_with_logo.png'
)
