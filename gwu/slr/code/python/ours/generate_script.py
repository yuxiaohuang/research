# Please cite the following paper when using the code


# Modules
import sys
import os


# Generate the script
def generate_script():
    # Get the script
    script = 'python' + ' ' + py_file + ' ' + data_file + ' ' + weight_file + ' ' + mse_fig + ' ' + prediction_file + ' ' + statistics_file + ' ' + probabilities_file + ' ' + max_iter + ' ' + C + ' ' + min_bin_size

    # Write the script file
    with open(script_file, 'w') as f:
        # Write the script
        f.write(script + '\n')


# Main function
if __name__=="__main__":
    # Get parameters from command line
    # Please see details of the parameters in the readme file
    py_file = sys.argv[1]
    script_dir = sys.argv[2]
    data_dir = sys.argv[3]
    weight_dir = sys.argv[4]
    mse_dir = sys.argv[5]
    prediction_dir = sys.argv[6]
    statistics_dir = sys.argv[7]
    probabilities_dir = sys.argv[8]
    max_iter = sys.argv[9]
    C = sys.argv[10]
    min_bin_size = sys.argv[11]

    # Make script_dir
    directory = os.path.dirname(script_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make weight_dir
    directory = os.path.dirname(weight_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make mes_dir
    directory = os.path.dirname(mse_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make prediction_dir
    directory = os.path.dirname(prediction_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make statistics_dir
    directory = os.path.dirname(statistics_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make probabilities_dir
    directory = os.path.dirname(probabilities_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for data_file in os.listdir(data_dir):
        if not data_file.startswith('.') and data_file.endswith(".txt"):
            # Update data_file
            data_file = data_dir + data_file

            # Get data file name
            data_file_name = os.path.basename(data_file)

            # Get script file
            script_file = script_dir + data_file_name

            # Get weight file
            weight_file = weight_dir + data_file_name

            # Get mse figure
            mse_fig = mse_dir + data_file_name.replace('.txt', '.pdf')

            # Get prediction file
            prediction_file = prediction_dir + data_file_name

            # Get statistics file
            statistics_file = statistics_dir + data_file_name

            # Get probabilities file
            probabilities_file = probabilities_dir + data_file_name

            # Generate the script
            generate_script()

