from flask import Flask, request, jsonify
from flask_cors import CORS
from GSOMPrediction import get_predictions_from_gsom, get_predictions_list_from_gsom
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

@app.route("/distressProbability", methods = ['POST'])
def distressPredictionProbability():
    file = request.files['audio']
    file.save('audio_recording.wav')
    predictions = get_predictions_from_cnn('audio_recording.wav')
    pred_list = get_predictions_list_from_gsom(predictions)
    num_zeros = pred_list.count(0)
    num_ones = pred_list.count(1)
    return jsonify({'prediction': [ round(num_zeros/len(pred_list),2),round(num_ones/len(pred_list),2) ]})


if __name__ == "__main__":
	app.run(host='0.0.0.0',port=5001)
