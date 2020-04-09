from flask import Flask, render_template, request
import model_cv
import pickle
import cv2
import numpy as np
import PIL
import matplotlib.pyplot as plt

MAX_FILE_SIZE = 1024 * 1024*10 + 1

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def index():
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        z = model_cv.predict (img)
        plt.imsave("./templates/new.jpg",z)
        #cv2.imwrite ("new.jpg",cv2.cvtColor(z, cv2.COLOR_BGR2RGB))
        if bool(file.filename):
            file_bytes = file.read(MAX_FILE_SIZE)
            args["file_size_error"] = len(file_bytes) == MAX_FILE_SIZE
        args["method"] = "POST"
    return render_template("index.html", args=args)

if __name__ == "__main__":
    app.run(debug=True)