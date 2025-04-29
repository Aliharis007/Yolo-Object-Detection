import cv2  # Import OpenCV for image processing
from ultralytics import YOLO  # Import YOLO from ultralytics for object detection

# Load the pre-trained YOLOv8 model (small variant)
model = YOLO("yolov8s.pt")  

def detect_objects(image_path):
    """
    Perform object detection on an image using YOLOv8.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None (Displays the image with detected objects)
    """
    
    # Read the image from the given path
    image = cv2.imread(image_path)

    # Perform object detection using YOLOv8
    results = model(image)

    # Iterate through detection results
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates (top-left and bottom-right)
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            
            # Extract confidence score of detection
            conf = box.conf[0].item()  
            
            # Extract class ID and convert to an integer
            cls = int(box.cls[0].item())  
            
            # Create label text with class name and confidence score
            label = f"{model.names[cls]}: {conf:.2f}"

            # Draw bounding box around detected object
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put the label text above the bounding box
            cv2.putText(image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detected objects
    cv2.imshow("Detected Objects", image)
    
    # Wait for a key press before closing the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function with a sample image
detect_objects("Ship2.jpg")  
