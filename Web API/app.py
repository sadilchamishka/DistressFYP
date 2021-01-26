from flask import Flask, request, jsonify
from flask_cors import CORS
from GSOMPrediction import get_predictions_from_gsom
from CNNPrediction import get_predictions_from_cnn

app = Flask(__name__)
cors = CORS(app)

@app.route("/", methods = ['POST'])
def distressPrediction():
    file = request.files['audio']
    file.save('audio_recording.wav')
    predictions = get_predictions_from_cnn('audio_recording.wav')
    distress_status = get_predictions_from_gsom(predictions)
    return jsonify({'status': distress_status})

if __name__ == "__main__":
	app.run(host='0.0.0.0')
