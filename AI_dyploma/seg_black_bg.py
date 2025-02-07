from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolo11m-seg.pt")
image_path = "1.jpg"
image = cv2.imread(image_path)

results = model(image_path, classes=[0])

output_image = np.zeros_like(image)

for result in results:
    masks = result.masks
    if masks is not None:
        for mask in masks:
            mask = mask.data[0].numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            output_image[mask > 0] = image[mask > 0]

cv2.imshow("Wynik segmentacji", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = "wynik_segmentacji.jpg"
cv2.imwrite(output_path, output_image)
print(f"Zapisano wynik do: {output_path}")