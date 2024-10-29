from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

def add_overlay_only(car_image_path, output_image_path, logo_path):
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
    if car_image is None:
        print("Error loading car image.")
        return

    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        print("Error loading logo image.")
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

            src_points = np.float32(box)

            overlay_w, overlay_h = int(rect[1][0]), int(rect[1][1])
            white_overlay = np.ones((overlay_h, overlay_w, 3), dtype=np.uint8) * 255
            dst_points_overlay = np.float32([
                [0, 0],
                [overlay_w, 0],
                [overlay_w, overlay_h],
                [0, overlay_h]
            ])

            matrix_overlay = cv2.getPerspectiveTransform(dst_points_overlay, src_points)
            warped_overlay = cv2.warpPerspective(white_overlay, matrix_overlay, (car_image.shape[1], car_image.shape[0]))

            mask_overlay = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
            _, mask_overlay = cv2.threshold(mask_overlay, 1, 255, cv2.THRESH_BINARY)
            car_image = cv2.bitwise_and(car_image, car_image, mask=cv2.bitwise_not(mask_overlay))
            car_image = cv2.add(car_image, warped_overlay)

            logo_resized = cv2.resize(logo, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)

            if logo_resized.shape[2] == 4:
                logo_mask = logo_resized[:, :, 3]
                logo_bgr = logo_resized[:, :, :3]
            else:
                logo_mask = np.ones((overlay_h, overlay_w), dtype=np.uint8) * 255
                logo_bgr = logo_resized

            roi = car_image[int(points[0][1]):int(points[0][1]) + overlay_h, int(points[0][0]):int(points[0][0]) + overlay_w]

            for c in range(0, 3):
                roi[:, :, c] = roi[:, :, c] * (1 - logo_mask / 255.0) + logo_bgr[:, :, c] * (logo_mask / 255.0)

            car_image[int(points[0][1]):int(points[0][1]) + overlay_h, int(points[0][0]):int(points[0][0]) + overlay_w] = roi

    cv2.imwrite(output_image_path, car_image)
    print(f"Output image saved at {output_image_path}")

add_overlay_only(
    car_image_path='test-images/car_3.png',
    output_image_path='output/cars_with_logo_overlay.png',
    logo_path='logo/Logo.png'
)
