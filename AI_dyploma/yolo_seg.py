from ultralytics import YOLO
import cv2
import numpy as np

# Wczytaj model YOLOv8 Segmentacji
model = YOLO("YOLO11m-seg.pt")

# Wczytaj obraz
image_path = "1.jpg"
image = cv2.imread(image_path)

# Wykonaj segmentację tylko dla ludzi (klasa 0)
results = model(image, classes=[0])

# Przetwarzaj wyniki
for result in results:
    # Pobierz maski segmentacji
    masks = result.masks  # Maski segmentacji

    if masks is not None:
        # Stwórz czarne tło
        black_background = np.zeros_like(image)  # Obraz o takim samym rozmiarze jak oryginalny, ale czarny

        # Przejdź przez wszystkie maski
        for i, mask in enumerate(masks):
            # Konwertuj maskę na obraz binarny (jednokanałowy)
            mask_np = mask.data.cpu().numpy()  # Pobierz maskę
            mask_np = (mask_np * 255).astype(np.uint8)  # Przeskaluj maskę do zakresu 0-255
            mask_np = np.squeeze(mask_np)  # Usuń dodatkowe wymiary, jeśli istnieją

            # Upewnij się, że maska jest w formacie CV_8U (jednokanałowy)
            if len(mask_np.shape) > 2:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)  # Konwertuj do skali szarości, jeśli to konieczne

            # Zmień rozmiar maski, aby pasował do rozmiaru obrazu
            mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

            # Skopiuj tylko obszar człowieka z oryginalnego obrazu na czarne tło
            person_on_black = cv2.bitwise_and(image, image, mask=mask_resized)

            # Dodaj wynikową osobę do czarnego tła
            black_background = cv2.add(black_background, person_on_black)

        # Wyświetl wynikowy obraz (wszyscy ludzie na czarnym tle)
        cv2.imshow("Ludzie na czarnym tle", black_background)
        cv2.waitKey(0)  # Czekaj na naciśnięcie klawisza
        cv2.destroyAllWindows()  # Zamknij okno po naciśnięciu klawisza

        # Zapisz wynikowy obraz (opcjonalnie)
        cv2.imwrite("ludzie_na_czarnym_tle.jpg", black_background)