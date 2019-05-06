
from flask import Flask
from flask import request
import os
import json
from brand_recog import init_detection,run_detection
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return "Hello World"

@app.route("/user")
def user():
    return "Neha Mittal"

@app.route("/find_brand", methods=['POST'])
def find_brand():
    print(request.files)
    if 'image' not in request.files:
        print 'No file part'
        return redirect(request.url)
    #image = request.form['name']
    image = request.files['image']
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    print(filename)
    brand = run_detection(filename)
    return brand

if __name__ == "__main__":
    init_detection()
    app.secret_key = os.urandom(12)
    app.run(debug=True, host='0.0.0.0', port=4000)
