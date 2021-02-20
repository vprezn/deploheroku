#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

cors = CORS(app, resources={r"/api/": {"origins": ""}})

# CORS Headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers',
                             'Content-Type,Authorization,true')
    response.headers.add('Access-Control-Allow-Methods',
                             'GET,PUT,POST,DELETE,OPTIONS')
    return response



#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/student_performance_prediction')
def predict():
    data = request.get_json()
    return jsonify({
                'success': True,
                'data': model.predit([
                            data["gender"],
                            data["Nationalty"],
                            data["place_of_birth"],
                            data["stage"],
                            data["grade"],
                            data["section"],
                            data["topic"],
                            data["semester"],
                            data["relation"],
                            data["raisedhands"],
                            data["visted_resource"],
                            data["AnnouncementsView"],
                            data["Discussion"],
                            data["ParentAnsweringSurvey"],
                            data["ParentschoolSatisfaction"],
                            data["StudentAbsenceDays"],
                        ]),
            })


if __name__ == "__main__":
    app.run(debug=True)
