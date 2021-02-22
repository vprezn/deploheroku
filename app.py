#import libraries
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
#Initialize the flask App
app = Flask(__name__)
cors = CORS(app, resources={r"/api/": {"origins": ""}})
 
# CORS Headers
@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response

student_performance_logistic_model = pickle.load(open('student_performance_logistic_model.pkl', 'rb'))


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

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


if __name__ == "__main__":
    app.run(debug=True)
