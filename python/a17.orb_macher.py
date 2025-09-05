import cv2
import numpy as np

def main():
    try:
        book = cv2.imread("data/book.jpg")
        if book is None:
            raise FileNotFoundError("Image file not found.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    book_gray = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)

    feature = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = feature.detectAndCompute(book_gray, None)

    # cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = feature.detectAndCompute(frame_gray, None)

    if des2 is not None:
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if matches:
            best_match = matches[0]
            print(f"Best match: {best_match.distance}")

    dst = cv2.drawMatches(book, kp1, frame, kp2, matches[:10], None, flags=2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    h, w = book.shape[:2]
    corners1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    if M is not None:
        corners2 = cv2.perspectiveTransform(corners1, M)
        frame = cv2.polylines(frame, [np.int32(corners2)], True, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Matches", dst)
    cv2.imshow("Book", book)
    cv2.imshow("Frame", frame)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
