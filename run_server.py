# import the necessary packages
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

# initialize Flask application
app = flask.Flask(__name__)

@app.route("/", methods=["POST","GET"])
def index():

    if flask.request.method == "GET":
        return flask.render_template('index.html')
    data = {}

    # load image
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
       
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)

            if preds[0,0] > 0.5:
                result ="Normal Image"
            else:
                result ="Abormal Image"                   
                       
            data["predictions result: "] = result            
          
    # return the data 
    return flask.jsonify(data)

def prepare_image(image, target):

    # if the image mode is not three channels, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32')
    image = image / 255
 
    return image

# start the server
if __name__ == "__main__":
    print(("* Flask starting server..."
        "please wait until server has fully started"))
    global model
    model = load_model('medical_diagnosis_cnn_model.h5')
    app.run(host='0.0.0.0')
    
    
    