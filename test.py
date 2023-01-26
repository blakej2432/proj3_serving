from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return "테스트 중 입니다."

@app.route("/test", methods=['POST'])
def translate():
    if request.method == 'POST':
        input_json = request.get_json()
        print(input_json['text'])
        if input_json['text'] == "받았니?":
            return "받았어"

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000', debug=True)
