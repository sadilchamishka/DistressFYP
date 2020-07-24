import time
import os
import joblib
import sys
sys.path.append('GSOM')

import data_parser as Parser
from util import utilities as Utils
from util import display as Display_Utils
from params import params as Params
from core4 import core_controller as Core

def GSOM_model(SF,forget_threshold,temporal_contexts,learning_itr,smoothing_irt,plot_for_itr,data_filename,output_save_location,name):

	# Init GSOM Parameters
	gsom_params = Params.GSOMParameters(SF, learning_itr, smoothing_irt, distance=Params.DistanceFunction.EUCLIDEAN,
										temporal_context_count=temporal_contexts, forget_itr_count=forget_threshold)
	generalise_params = Params.GeneraliseParameters(gsom_params)

	# Process the input files
	input_vector_database, labels, classes = Parser.InputParser.parse_input_train_data(data_filename, None)

	# Setup the age threshold based on the input vector length
	generalise_params.setup_age_threshold(input_vector_database[0].shape[0])

	# Process the clustering algorithm 
	controller = Core.Controller(generalise_params)
	controller_start = time.time()
	result_dict = controller.run(input_vector_database, plot_for_itr, classes)
	print('Algorithms completed in', round(time.time() - controller_start, 2), '(s)')

	gsom_nodemap = result_dict[0]['gsom']

	# Saving gsom node map
	saved_gsom_nodemap_for_0_7 = joblib.dump(gsom_nodemap, output_save_location+'gsom_nodemap_{}.joblib'.format(name))

	# Display
	display = Display_Utils.Display(result_dict[0]['gsom'], None)
	display.setup_labels_for_gsom_nodemap(classes, 2, 'Latent Space of cnn_5100_input_file_to_gsom : SF=0.7',output_save_location+'latent_space_{}_hitvalues'.format(name))
	print('Completed.')