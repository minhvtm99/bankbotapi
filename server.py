from flask import Flask, request, jsonify
from flask_restful import Resource, Api

import chatbot
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
    # incoming_msg = request.json['message']
    # tagged_msg = chatbot.tag_experiment(incoming_msg, 2)

    # Tra ve ket qua dang json
    # return jsonify(categorized_msg = tagged_msg)
    return jsonify(request.json)

if __name__ == '__main__':
    app.run()

