import pandas as pd

class InputParser:

    @staticmethod
    def parse_input_train_data(filename, header='infer'):

        input_data = pd.read_csv(filename, header=header)

        classes = input_data[513].tolist()
        labels = input_data[0].tolist()
        input_database = {
            0: input_data.values[:,1:-1]
        }

        return input_database, labels, classes

    @staticmethod
    def parse_input_test_data(filename, header=None):

        input_data = pd.read_csv(filename, header=header)


        labels = input_data[0].tolist()
        input_database = {
            0: input_data.values[:,1:]
        }

        return input_database, labels