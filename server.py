from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps

from Documents.GitHub.bankbotapi.chatbot import tag_experiment
# from flask.ext.jsonpify import jsonify

app = Flask(__name__)
api = Api(app)


@app.route('/message_categorize', methods=['POST']) 
def message_categorize(): 
    # nhan ve chuoi request.json, dang: 
       
    # {
    #     "message":"Toi muon tim atm"
    # }

    # Xu ly parse o day
    incoming_msg = request.json['message']
    tagged_msg = tag_experiment(incoming_msg, 2)

    # Tra ve ket qua dang json
    return jsonify(categorized_msg = tagged_msg)


if __name__ == '__main__':
     app.run()

