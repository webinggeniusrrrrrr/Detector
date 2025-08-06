from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load models
age_net = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@app.route('/', methods=['GET', 'POST'])
def index():
    age_result = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = 'static/input.jpg'
            file.save(filepath)

            image = cv2.imread(filepath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_img = image[y:y + h, x:x + w]
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                             (78.426, 87.768, 114.895), swapRB=False)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age_result = AGE_BUCKETS[age_preds[0].argmax()]

                label = f"Age: {age_result}"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imwrite("static/result.jpg", image)

    return render_template('index.html', age=age_result)

if __name__ == '__main__':
    app.run(debug=True)
