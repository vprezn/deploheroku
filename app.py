#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/student_performance_prediction')
def predict():
    '''
    For rendering results on HTML GUI
    '''

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
