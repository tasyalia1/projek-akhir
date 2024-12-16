from flask import Flask, render_template, request, Response
from ultralytics import YOLO
from PIL import Image
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLO model once to optimize performance
model = YOLO("best (3).pt")

@app.route("/dashboard")
def dashboard(): 
    return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template("upload.html")

    elif request.method == 'POST':
        action = request.form.get('action')
        
        if action == "image":
            # Handle image detection
            f = request.files['img']
            img = Image.open(f.stream)

            # Predict image
            result = model.predict(img)
            result[0].save(os.path.join(app.config['UPLOAD_FOLDER'], "deteksi.jpg"))

            return render_template("upload.html", predicted_image=True)
        
        elif action == "video":
            # Handle video detection
            f = request.files['video']
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_video.mp4")
            f.save(video_path)
            
            # Redirect to video streaming route
            return render_template("upload.html", video_stream=True)

@app.route("/video_feed")
def video_feed():
    def generate_frames():
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_video.mp4")
        cap = cv2.VideoCapture(video_path)
        
        target_width = 640  # Lebar target
        target_height = 360 # Tinggi target (atau sesuaikan dengan proporsi)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Predict and draw results
            results = model.predict(frame_rgb)
            annotated_frame = results[0].plot()
            
            # Convert back to BGR for OpenCV compatibility
            frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=5001, debug=True)
