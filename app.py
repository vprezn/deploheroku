#import libraries 
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
# import cv2
# import tensorflow as tf
# import tensorflow_hub as hub
import librosa
import sklearn
from PIL import Image

#Initialize the flask App
app = Flask(__name__)
cors = CORS(app, resources={r"/api/": {"origins": ""}})
# cors = CORS(app)
 
# CORS Headers
@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response



student_performance_logistic_model = pickle.load(open('student_performance_logistic_model.pkl', 'rb'))
land_price_prediction_ridge_model = pickle.load(open('land_price_prediction_ridge_model.pkl', 'rb'))
randomeforest_model = pickle.load(open('randomforest_model.pkl', 'rb'))
# model = load_model('keras_model.h5')
model = ""
batch_size = 64
image_size = 224


class_names = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue",
 "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers",
  "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy",
   "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus",
    "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily",
     "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily",
      "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea",
       "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation",
        "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower",
         "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort",
          "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil",
           "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum",
            "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold",
             "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose",
              "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya",
               "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily",
                "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower",
                 "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia",
                  "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea",
                   "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff",
                    "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea",
                     "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", 
                     "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}

#default page of our web-app
@app.route('/')
def home():
    return "Ahmad"

#To use the predict button in our web-app
@app.route('/student_performance_prediction',methods=['GET','POST'])
def predict():
    if request.method == "POST":
        data = request.get_json()
        return jsonify({ 
            'success': True,
            'data': '{}'.format(student_performance_logistic_model.predict([[
                int(data["gender"]),
                int(data["Nationalty"]),
                int(data["place_of_birth"]),
                int(data["stage"]),
                int(data["grade"]),
                int(data["section"]),
                int(data["topic"]),
                int(data["semester"]),
                int(data["relation"]),
                int(data["raisedhands"]),
                int(data["visted_resource"]),
                int(data["AnnouncementsView"]),
                int(data["Discussion"]),
                int(data["ParentAnsweringSurvey"]),
                int(data["ParentschoolSatisfaction"]),
                int(data["StudentAbsenceDays"])
            ]])[0].item())
            })


@app.route('/land_prices_prediction',methods=['GET','POST'])
def predict_land():
    if request.method == "POST":
        data = request.get_json()
        pred = [int(data['streetwidth']),int(data['size']),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        ,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        pred[int(data['MainLocation'])] = 1.0
        pred[int(data['SubLocation'])] = 1.0
        pred[int(data['Neighborhood'])] = 1.0
        pred[int(data['frontage'])] = 1.0
        pred[int(data['purpose'])] = 1.0
        return jsonify({ 
            'success': True,
            'data': '{}'.format(land_price_prediction_ridge_model.predict([pred])[0].item())
            })


@app.route('/gender_prediction',methods=['GET','POST'])
def predict_gender():
    if request.method == "POST":
        if 'audio_file' not in request.files:
            return jsonify({'success': True,'data': 'false'})
        
        features_label = np.load('features_label.npy', allow_pickle=True)
        features = []
        for i in range(0, len(features_label)):
            features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                        features_label[i][2], features_label[i][3],
                        features_label[i][4]), axis=0))

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit_transform(np.array(features))
        feat = extract_features(request.files.get('audio_file'))
        features = np.concatenate((feat[0], feat[1], feat[2], feat[3],feat[4]), axis=0)
        c = scaler.transform([features])
        prediction = randomeforest_model.predict(c)
        return jsonify({ 'success': True,'data': '{}'.format(prediction[0])})
            
    return jsonify({'success': True,'data': False})


@app.route('/flower_prediction',methods=['POST'])
def predict_flower():
    if request.method == "POST":
        im = Image.open(request.files.get('file'))
        model_name = "flower_twnsorflow.h5"
        image_path = im
        model = tf.keras.models.load_model(model_name ,custom_objects={'KerasLayer':hub.KerasLayer} )
        top_k = 3
        print("Loaded")
        probs, classes = predict_flower_image(image_path, model, top_k)
        return jsonify({ 'success': True,'data': [probs,classes]})
            
    return jsonify({'success': True,'data': False})


def process_image(img):
    tf_img = tf.image.convert_image_dtype(img, dtype=tf.int16, saturate=False)
    tf_img = tf.image.resize(img,(224,224)).numpy()
    tf_img = tf_img/255

    return tf_img


def predict_flower_image(image_path,model,top_k = 3):
    img     = np.asarray(image_path)
    pro_img = process_image(img)
    expanded_img = model.predict(np.expand_dims(pro_img, axis=0))
    values, indices= tf.nn.top_k(expanded_img, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0] + 1
    flowers = []
    for flower in class_names:
        if str(classes[0]) == flower:
            flowers.insert(0,class_names[flower])
        if str(classes[1]) == flower:
            flowers.insert(1,class_names[flower])
        if str(classes[2]) == flower:
            flowers.insert(2,class_names[flower])
    
    pr = []
    for i in range(len(probs)):
        pr.append(format(probs[i],'f'))
    return pr,flowers

def extract_features(files):
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(files, res_type='kaiser_fast') 

    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)


    return mfccs, chroma, mel, contrast, tonnetz




# Oktoub
@app.route('/predict',methods=['POST'])
def predict():
    
    imgBytes = request.get_json()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # preprocessing
    res = preprocessing(imgBytes)
    

    #turn the image into a numpy array
    image_array = np.asarray(res)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    prediction = model.predict(data).argmax()
    
    return jsonify({
        "success":True,
        "pred":str(prediction)
    })

def preprocessing(data):
    data = np.asarray(data, dtype = np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image

    constant= cv2.copyMakeBorder(rect,20,20,20,20,cv2.BORDER_CONSTANT,value=[255,255,255])
    return cv2.resize(constant, dsize=(224, 224))

if __name__ == "__main__":
    app.run(debug=True)
