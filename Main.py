import os
from dotenv import load_dotenv
import cv2
import time
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
from ultralytics import YOLO
import threading

# Load environment variables from .env file
load_dotenv()

# Retrieve the SendGrid API key and email details from environment variables
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

def draw_text_with_background(frame, text, position, font_scale=0.4, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), alpha=0.7, padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    overlay = frame.copy()
    x, y = position
    cv2.rectangle(overlay, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def send_email_alert(image_path):
    subject = "Alert: Hardhat Missing!"
    body = "A hardhat was not detected for the past 10 seconds, but a person was detected. Please find the attached frame showing the situation."

    # Read the image file and encode it
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()

    # Create email message
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=RECEIVER_EMAIL,
        subject=subject,
        plain_text_content=body
    )

    # Attach image to the email
    attachment = Attachment(
        FileContent(encoded_image),
        FileName(image_path),
        FileType("image/jpeg"),
        Disposition("attachment")
    )
    message.attachment = attachment

    # Send email using SendGrid
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"Email sent! Status Code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_email_in_background(image_path):
    email_thread = threading.Thread(target=send_email_alert, args=(image_path,))
    email_thread.start()

def main():
    model = YOLO("Model/sbai.pt")  # Replace with your custom model file if needed
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to exit.")

    colors = [
        (255, 0, 0),  # Hardhat (Blue)
        (0, 255, 0),  # Mask (Green)
        (0, 0, 255),  # NO-Hardhat (Red)
        (255, 255, 0),  # NO-Mask (Cyan)
        (255, 0, 255),  # NO-Safety Vest (Magenta)
        (0, 255, 255),  # Person (Yellow)
        (128, 0, 128),  # Safety Cone (Purple)
        (128, 128, 0),  # Safety Vest (Olive)
        (0, 128, 128),  # Machinery (Teal)
        (128, 128, 128)  # Vehicle (Gray)
    ]

    last_hardhat_time = time.time()
    last_email_time = time.time()
    email_sent_flag = False
    email_sent_time = 0

    cv2.namedWindow("YOLOv8 Annotated Feed", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break

        hardhat_detected = False
        person_detected = False

        results = model(frame)

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    cls = int(box.cls[0])
                    label = f"{model.names[cls]} ({confidence:.2f})"

                    color = colors[cls % len(colors)]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_text_with_background(frame, label, (x1, y1 - 10), font_scale=0.4, color=(255, 255, 255), bg_color=color, alpha=0.8, padding=4)

                    if model.names[cls] == "Hardhat":
                        hardhat_detected = True
                    elif model.names[cls] == "Person":
                        person_detected = True

        if person_detected and not hardhat_detected and (time.time() - last_email_time) >= 100:
            image_path = "no_hardhat_frame.jpg"
            cv2.imwrite(image_path, frame)
            send_email_in_background(image_path)
            email_sent_flag = True
            email_sent_time = time.time()
            last_email_time = time.time()

        if hardhat_detected:
            last_hardhat_time = time.time()

        if email_sent_flag and (time.time() - email_sent_time) < 3:
            draw_text_with_background(frame, "Email Sent", (frame.shape[1] - 100, 30), font_scale=0.5, color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.8, padding=5)

        resized_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("YOLOv8 Annotated Feed", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
