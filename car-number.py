from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

def add_rotated_logo_with_overlay(car_image_path, logo_image_path, output_image_path):
    CLIENT = InferenceHTTPClient(
        api_url="https://outline.roboflow.com",
        api_key="mb7m4Bcuxtbp5gAuPR0j"
    )

    response = CLIENT.infer(car_image_path, model_id="masking-license-plates/1")
    predictions = response.get("predictions", [])
    if not predictions:
        print("No license plates detected in the image.")
        return

    car_image = cv2.imread(car_image_path)
    logo_image = cv2.imread(logo_image_path, cv2.IMREAD_UNCHANGED)
    if car_image is None or logo_image is None:
        print("Error loading images.")
        return

    for pred in predictions:
        if pred["class"] == "plate":
            points = np.array([(int(point["x"]), int(point["y"])) for point in pred["points"]])

            if len(points) < 4:
                print("Not enough points to define corners.")
                continue

            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            src_points = np.float32([
                box[1],
                box[2],
                box[3],
                box[0]
            ])

            h, w = car_image.shape[:2]
            white_overlay = np.ones((h, w, 3), dtype=np.uint8) * 255
            dst_points_overlay = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ])

            matrix_overlay = cv2.getPerspectiveTransform(dst_points_overlay, src_points)
            warped_overlay = cv2.warpPerspective(white_overlay, matrix_overlay, (car_image.shape[1], car_image.shape[0]))

            mask_overlay = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
            _, mask_overlay = cv2.threshold(mask_overlay, 1, 255, cv2.THRESH_BINARY)
            car_image = cv2.bitwise_and(car_image, car_image, mask=cv2.bitwise_not(mask_overlay))
            car_image = cv2.add(car_image, warped_overlay)

            logo_h, logo_w = logo_image.shape[:2]
            dst_points_logo = np.float32([
                [0, 0],
                [logo_w, 0],
                [logo_w, logo_h],
                [0, logo_h]
            ])

            matrix_logo = cv2.getPerspectiveTransform(dst_points_logo, src_points)
            warped_logo = cv2.warpPerspective(logo_image, matrix_logo, (car_image.shape[1], car_image.shape[0]))

            if warped_logo.shape[2] == 4:
                alpha_mask = warped_logo[:, :, 3] / 255.0
                for c in range(3):
                    car_image[:, :, c] = (1 - alpha_mask) * car_image[:, :, c] + alpha_mask * warped_logo[:, :, c]
            else:
                mask_logo = cv2.cvtColor(warped_logo, cv2.COLOR_BGR2GRAY)
                _, mask_logo = cv2.threshold(mask_logo, 1, 255, cv2.THRESH_BINARY)
                car_image = cv2.bitwise_and(car_image, car_image, mask=cv2.bitwise_not(mask_logo))
                car_image = cv2.add(car_image, warped_logo)

    cv2.imwrite(output_image_path, car_image)
    print(f"Output image saved at {output_image_path}")

add_rotated_logo_with_overlay(
    car_image_path='test-images/car_3.png',
    logo_image_path='logo/Logo.png',
    output_image_path='output/cars_with_rotated_logo_and_overlay.png'
)
