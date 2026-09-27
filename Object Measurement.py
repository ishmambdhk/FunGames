mport cv2  # Import OpenCV for video stream capture, image processing, and visualization
import utlis  # Import custom utility module (contains contour detection, perspective warping, point ordering)

###################################
webcam = True  # Flag setting: Set to True for live camera input, False to read a static image file
path = '4.jpg'  # Fallback image path used when webcam flag is False
cap = cv2.VideoCapture(0)  # Initialize video capture using default camera index 0
cap.set(10, 160)  # Set camera brightness property ID 10 (CAP_PROP_BRIGHTNESS) to 160
cap.set(3, 1920)  # Set camera frame width property ID 3 (CAP_PROP_FRAME_WIDTH) to 1920 pixels
cap.set(4, 1080)  # Set camera frame height property ID 4 (CAP_PROP_FRAME_HEIGHT) to 1080 pixels
scale = 3  # Multiplication factor to upscale A4 paper pixel dimensions for higher measurement resolution
wP = 210 * scale  # Scaled target width for A4 reference sheet (210mm real width * 3 = 630 pixels)
hP = 297 * scale  # Scaled target height for A4 reference sheet (297mm real height * 3 = 891 pixels)
###################################

while True:  # Continuous frame processing loop
    if webcam:
        success, img = cap.read()  # Grab live frame from camera stream (success status boolean, img array)
    else:
        img = cv2.imread(path)  # Read static image file from disk path

    # Detect primary contours (A4 paper sheet) filtering for minimum area 50,000 and 4 corners
    imgContours, conts = utlis.getContours(img, minArea=50000, filter=4)

    if len(conts) != 0:  # Check if at least one large 4-sided reference object (A4 paper) was found
        biggest = conts[0][2]  # Extract corner coordinates of the largest detected contour (sorted by area)
        # print(biggest)                         # Debug line to print raw corner coordinate array

        # Flatten/correct perspective of the detected A4 sheet into a top-down view of size wP x hP
        imgWarp = utlis.warpImg(img, biggest, wP, hP)

        # Detect smaller objects resting on top of the warped top-down A4 image sheet
        imgContours2, conts2 = utlis.getContours(
            imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False
        )

        if (
            len(conts) != 0
        ):  # Check if objects exist (Note: typical code bug here; should check len(conts2))
            for obj in conts2:  # Loop over each detected object contour on the A4 paper
                cv2.polylines(
                    imgContours2, [obj[2]], True, (0, 255, 0), 2
                )  # Draw green boundary around object (thickness 2)
                nPoints = utlis.reorder(
                    obj[2]
                )  # Sort 4 corner points in standard order: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]

                # Calculate physical width (mm converted to cm): distance between TL & TR / scale / 10
                nW = round(
                    (
                        utlis.findDis(
                            nPoints[0][0] // scale, nPoints[1][0] // scale
                        )
                        / 10
                    ),
                    1,
                )

                # Calculate physical height (mm converted to cm): distance between TL & BL / scale / 10
                nH = round(
                    (
                        utlis.findDis(
                            nPoints[0][0] // scale, nPoints[2][0] // scale
                        )
                        / 10
                    ),
                    1,
                )

                # Draw dimension arrow line along top horizontal edge (Width) in magenta color
                cv2.arrowedLine(
                    imgContours2,
                    (nPoints[0][0][0], nPoints[0][0][1]),
                    (nPoints[1][0][0], nPoints[1][0][1]),
                    (255, 0, 255),
                    3,
                    8,
                    0,
                    0.05,
                )

                # Draw dimension arrow line along left vertical edge (Height) in magenta color
                cv2.arrowedLine(
                    imgContours2,
                    (nPoints[0][0][0], nPoints[0][0][1]),
                    (nPoints[2][0][0], nPoints[2][0][1]),
                    (255, 0, 255),
                    3,
                    8,
                    0,
                    0.05,
                )

                x, y, w, h = obj[
                    3
                ]  # Unpack bounding box metrics (x-coordinate, y-coordinate, width, height)

                # Overlay calculated width text in cm above the object bounding box
                cv2.putText(
                    imgContours2,
                    "{}cm".format(nW),
                    (x + 30, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (255, 0, 255),
                    2,
                )

                # Overlay calculated height text in cm to the left side of the object bounding box
                cv2.putText(
                    imgContours2,
                    "{}cm".format(nH),
                    (x - 70, y + h // 2),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (255, 0, 255),
                    2,
                )

        cv2.imshow(
            "A4", imgContours2
        )  # Render warped top-down image window with object measurements

    img = cv2.resize(
        img, (0, 0), None, 0.5, 0.5
    )  # Downscale raw webcam image to half resolution (50%) for monitor display
    cv2.imshow(
        "Original", img
    )  # Render scaled original webcam view in window titled 'Original'
    cv2.waitKey(
        1
    )  # Pause execution 1ms to process display window events and refresh UI
