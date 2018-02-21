# Please cite the following paper when using the code


# Modules
import sys
import os


# Generate the script
def generate_script():
    # Get the script
    script = 'python' + ' ' + py_file + ' ' + data_file + ' ' + statistics_file

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
    statistics_dir = sys.argv[4]

    # Make script_dir
    directory = os.path.dirname(script_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make statistics_dir
    directory = os.path.dirname(statistics_dir)
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

            # Get statistics file
            statistics_file = statistics_dir + data_file_name

            # Generate the script
            generate_script()

