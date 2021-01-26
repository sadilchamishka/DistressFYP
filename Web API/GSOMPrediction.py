import joblib
import sys

sys.path.append('GSOM')
import data_parser as Parser
from util import utilities as Utils
from params import params as Params
from core4 import core_controller as Core

gsom_nodemap_5000_for_0_5 = joblib.load('gsom_nodemap_5100_for_0_7.joblib')
threshold=0.35

def get_predictions_from_gsom(predictions):

    f = open("input_file_to_gsom","w")
    for i in range(len(predictions)):
        comma_separated_features = ','.join([str(i) for i in predictions[i]])
        f.write(str(i) + "," + comma_separated_features +"\n")   

    input_vector_database, labels = Parser.InputParser.parse_input_test_data('input_file_to_gsom', None) 

    test_predictions = []
    for input_id in labels:
        winner = Utils.Utilities.select_winner(gsom_nodemap, input_vector_database[0][input_id], Params.DistanceFunction.EUCLIDEAN, -1)
        label_list = get_winner_labels(gsom_nodemap,winner.x,winner.y)

        radius=0
        x = winner.x
        y = winner.y

        while(len(label_list)==0):      
            radius=radius+1
            label_list=label_list + get_labels_in_radius(gsom_nodemap,radius,x,y)
        
        a = label_list.count('1')
        b = label_list.count('0')
        p = a/(a+b)

        if p>=threshold:
            test_predictions.append(1)
        else:
            test_predictions.append(0)

    return int(np.round(np.average(test_predictions)))


def get_labels_in_radius(gsom_nodemap,radius,x,y):  
  label_list = get_winner_labels(gsom_nodemap,x+radius,y)+get_winner_labels(gsom_nodemap,x-radius,y)+get_winner_labels(gsom_nodemap,x,y+radius)+get_winner_labels(gsom_nodemap,x,y-radius)+get_winner_labels(gsom_nodemap,x+radius,y+radius)+get_winner_labels(gsom_nodemap,x+radius,y-radius)+get_winner_labels(gsom_nodemap,x-radius,y+radius)+get_winner_labels(gsom_nodemap,x-radius,y-radius)
  return label_list

def get_winner_labels(gsom_nodemap,n,m):
  winner_key = Utils.Utilities.generate_index(n, m)
  try:
    mapped_input_labels = gsom_nodemap[winner_key].get_mapped_labels()
    return [str(classes[lbl_id]) for lbl_id in mapped_input_labels]

  except KeyError:
    #if the new generated key does not exist in the original key map
    return []