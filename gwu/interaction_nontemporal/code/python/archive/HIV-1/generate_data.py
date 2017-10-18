

# Please cite the following paper when using the code


# Modules
import sys
import os
import csv

# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list

# Global variables:
amino_acid_L = ['G', 'P', 'A', 'V', 'L', 'I', 'M', 'C', 'F', 'Y', 'W', 'H', 'K', 'R', 'Q', 'N', 'E', 'D', 'S', 'T']
y = "class_-1"

# The dictionary of value
# key: time->var
# val: value of var at the time
val_Dic = {}

# Maximum time stamp
max_time_stamp = 0


# Generate source and target data
def generate_data():
    # Get the source list
    x_L = []
    for i in range(8):
        for amino_acid in amino_acid_L:
            x = str(i) + '_' + amino_acid
            x_L.append(x)

    # Load the raw data file
    with open(raw_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter=','))

        global max_time_stamp
        # Get the maximum time stamp
        max_time_stamp = len(spamreader)

        # From the first line to the last (since there is no header)
        for i in range(0, max_time_stamp):
            # Initialization
            if not i in val_Dic:
                val_Dic[i] = {}

            # Get the source value
            octamers = spamreader[i][0]
            for x in x_L:
                for j in range(8):
                    amino_acid = octamers[j]
                    if str(j) in x and amino_acid in x:
                        val_Dic[i][x] = 1
                        break
                if not x in val_Dic[i]:
                    val_Dic[i][x] = 0

            # Get the target value
            HIV_1_protease_cleavage = spamreader[i][1]
            if HIV_1_protease_cleavage == '-1':
                val_Dic[i][y] = 1
            else:
                val_Dic[i][y] = 0

    write_file(src_data_training_file, 'src', 'training')
    write_file(src_data_testing_file, 'src', 'testing')
    write_file(tar_data_training_file, 'tar', 'training')
    write_file(tar_data_testing_file, 'tar', 'testing')


# Write file
def write_file(file, src_tar_F, training_testing_F):
    # Write file
    with open(file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Get the header
        header_L = []
        for name_val in sorted(val_Dic[0].keys()):
            if ((src_tar_F == 'src' and not 'class' in name_val)
                or (src_tar_F == 'tar' and 'class' in name_val)):
                header_L.append(name_val)

        # Write the header
        spamwriter.writerow(header_L)

        # Write the value
        # Get start and end
        if training_testing_F == 'training':
            start = 0
            end = int(0.8 * max_time_stamp)
        else:
            start = int(0.8 * max_time_stamp)
            end = max_time_stamp

        # Get iteration
        if training_testing_F == 'training':
            iteration = 1
        else:
            iteration = 1

        for i in range(iteration):
            for time in range(start, end):
                # The list of value at the time
                val_L = []

                if time in val_Dic:
                    for name_val in header_L:
                        if name_val in val_Dic[time]:
                            val_L.append(val_Dic[time][name_val])
                        else:
                            val_L.append(0)
                else:
                    for name_val in src_L:
                        val_L.append(0)

                spamwriter.writerow(val_L)


# Main function
if __name__ == "__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_file_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]

    # Make directory
    directory = os.path.dirname(src_data_dir + '/training/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(src_data_dir + '/testing/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/training/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir + '/testing/')
    if not os.path.exists(directory):
        os.makedirs(directory)

    for raw_file in os.listdir(raw_file_dir):
        if raw_file.endswith(".txt"):
            # Get src data training file
            src_data_training_file = src_data_dir + '/training/src_data_' + raw_file
            # Get tar data training file
            tar_data_training_file = tar_data_dir + '/training/tar_data_' + raw_file

            # Get src data testing file
            src_data_testing_file = src_data_dir + '/testing/src_data_' + raw_file
            # Get tar data testing file
            tar_data_testing_file = tar_data_dir + '/testing/tar_data_' + raw_file

            # Update raw_file
            raw_file = raw_file_dir + raw_file

            # Generate data
            generate_data()

