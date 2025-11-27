# thanks to Martin Breuss: https://realpython.com/python-web-applications/
# thanks to Dipal Bhavsar: https://www.bacancytechnology.com/blog/react-with-python

from flask import Flask, request, jsonify
from flask_cors import CORS # cross-origin resource sharing (one domain requests a resource from another domain; frontend communicates w/ backend)
import json
import pandas as pd

app = Flask(__name__)
CORS(app=app)

@app.route('/', methods=['GET'])
def get_data():
    # thanks to Tim Santeford: https://www.timsanteford.com/posts/how-to-read-and-parse-jsonl-files-in-python/
    file_path = "national_capitals.jsonl"
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return data

from flask import send_from_directory

@app.route('/')
def serve():
    return send_from_directory('react-frontend/build', 'index.html')

# @app.route("/")
# def index():
#     celsius = request.args.get("celsius", "")
#     if celsius:
#         fahrenheit = fahrenheit_from(celsius)
#     else:
#         fahrenheit = ""
#     return (
#         """<form action="" method="get">
#                 Celsius temperature: <input type="text" name="celsius">
#                 <input type="submit" value="Convert to Fahrenheit">
#             </form>"""
#         + "Fahrenheit: "
#         + fahrenheit
#     )

# def fahrenheit_from(celsius):
#     """Convert Celsius to Fahrenheit degrees."""
#     try:
#         fahrenheit = float(celsius) * 9 / 5 + 32
#         fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
#         return str(fahrenheit)
#     except ValueError:
#         return "invalid input"

if __name__ == "__main__":
    # app.run(host="127.0.0.1", port=8080, debug=True)
    app.run(debug=True)