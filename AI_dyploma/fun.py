from ultralytics import YOLO
import cv2

# Załaduj model
model = YOLO("yolo11m-seg.pt")  # Załaduj wcześniej wytrenowany model

# Wczytaj obraz
image_path = "1.jpg"
image = cv2.imread(image_path)

# Zastosowanie modelu na obrazie
results = model(image_path, classes=[0])  # Przeszukaj obraz tylko pod kątem klasy 0 (można zmienić na inne klasy)

# Inicjalizacja wyników w odpowiednim formacie
output_data = {
    "image": cv2.imread(image_path),  # Zapisujemy obraz w oryginalnej formie
    "bboxes": [],  # Lista do przechowywania bounding boxów
    "name": image_path  # Przechowujemy nazwę obrazu
}

# Iteracja przez wszystkie wyniki (detekcje) na obrazie
for result in results:
    # Wykonaj detekcję i narysuj bounding boxy na obrazie
    annotated_image = result.plot()  

    # Pobierz współrzędne bounding boxów
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            # (x_center, y_center, width, height)
            x_center, y_center, width, height = box.xywh[0]  # współrzędne bounding boxa (x_center, y_center, w, h)
            confidence = box.conf[0]  # pewność detekcji
            class_id = int(box.cls[0].item())  # Konwersja na int
            
            # Konwersja wartości tensorowych na float
            x_center = float(x_center.item())
            y_center = float(y_center.item())
            width = float(width.item())
            height = float(height.item())

            # Oblicz współrzędne lewego górnego rogu (x1, y1)
            x1 = round(x_center - width / 2, 2)
            y1 = round(y_center - height / 2, 2)

            # Dodaj bounding box do listy wyników w wymaganym formacie
            output_data["bboxes"].append((x1, y1, width, height))

            # Rysowanie bounding boxa na obrazie
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x1 + width), int(y1 + height)), (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Confidence: {confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Wyświetl wynik detekcji
    cv2.imshow("Wynik detekcji", annotated_image)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  

    # Zapisz wynik do pliku
    output_path = "wynik_detekcji.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Zapisano wynik do: {output_path}")

# Zwróć dane wyjściowe w pożądanym formacie
print(f"Wynik w formacie: {output_data}")
