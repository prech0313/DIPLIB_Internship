import diplib as dip
import numpy as np
import cv2

# Load the image using OpenCV
image_path = r'C:\Users\salon\work\cmm.jpg'
image_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image_cv is None:
    print("Error: Image not loaded correctly.")
    exit()

# Convert the OpenCV image to a DIPlib image
dip_image = dip.Image(image_cv.astype(np.float32))

# Apply some DIPlib processing (e.g., Gaussian smoothing)
dip_image_smoothed = dip.Gauss(dip_image, 2.0)

# Convert the DIPlib image back to a NumPy array for use with OpenCV
image_np = np.array(dip_image_smoothed, dtype=np.uint8)

# Detect circles using Hough Circle Transform in OpenCV
circles = cv2.HoughCircles(
    image_np,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=5,
    maxRadius=50
)

# Ensure some circles have been found
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    # Extract the centers of the circles
    centers = [(x, y) for x, y, r in circles]

    # Create the output image for displaying results
    output_image = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2BGR)

    # Draw the detected circles and number them
    for idx, (x, y, r) in enumerate(circles, start=1):
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)
        cv2.putText(output_image, f"{idx}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate the distance between all pairs of circles using DIPlib
    num_circles = len(centers)
    for i in range(num_circles):
        for j in range(i + 1, num_circles):
            point1 = np.array([centers[i][0], centers[i][1]])
            point2 = np.array([centers[j][0], centers[j][1]])

            point1_dip = dip.Image(point1.astype(np.float32))
            point2_dip = dip.Image(point2.astype(np.float32))

            squared_diff = (point1_dip - point2_dip)**2
            sum_squared_diff = dip.Sum(squared_diff)

            distance = np.sqrt(sum_squared_diff)
            distance = distance[0]  # Extract scalar from DIPlib's Image object
            distance_mm = distance  # Initially 1:1 ratio

            print(f"Distance between circle {i + 1} and circle {j + 1}: {distance_mm:.2f} mm")

            # Draw the line between the circles and annotate the distance
            cv2.line(output_image, tuple(point1), tuple(point2), (255, 0, 0), 2)
            mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
            cv2.putText(output_image, f"{distance_mm:.2f} mm", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the output image
    cv2.imshow("Circles and Distances", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No circles detected.")
