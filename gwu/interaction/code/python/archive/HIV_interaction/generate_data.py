

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
y = "HIV_1_protease_cleavage"

# The dictionary of value
# key: time->var
# val: value of var at the time
val_Dic = {}


# Generate source and target data
def generate_data():
    # Get the source list
    x_L = []
    for i in range(8):
        for amino_acid in amino_acid_L:
            x = str(i) + '_' + amino_acid
            x_L.append(x)

    # Time points, -1 by default
    time = -1
    # Load the raw data file
    with open(raw_data_file, 'r') as f:
        spamwriter = csv.reader(f, delimiter = ',')

        for line in spamwriter:
            time += 1
            # Initialization
            if not time in val_Dic:
                val_Dic[time] = {}

            # Get the source value
            octamers = line[0]
            for x in x_L:
                for i in range(8):
                    amino_acid = octamers[i]
                    if str(i) in x and amino_acid in x:
                        val_Dic[time][x] = 1
                        break
                if not x in val_Dic[time]:
                    val_Dic[time][x] = 0

            # Get the target value
            if time == 0:
                val_Dic[0][y] = 0

            if not (time + 1) in val_Dic:
                # Initialization
                val_Dic[time + 1] = {}

            HIV_1_protease_cleavage = line[1]
            if HIV_1_protease_cleavage == '1':
                val_Dic[time + 1][y] = 1
            else:
                val_Dic[time + 1][y] = 0


    # Write the source file
    with open(src_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter = ',')
        # Write the header
        spamwriter.writerow(x_L)

        # Write the value
        for time in sorted(val_Dic.keys()):
            val_L = []
            for x in x_L:
                val = 0
                if x in val_Dic[time]:
                    val = val_Dic[time][x]
                val_L.append(val)

            spamwriter.writerow(val_L)

    # Write the target file
    with open(tar_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')
        # Write the header
        spamwriter.writerow([y])

        # Write the value
        for time in val_Dic:
            val = 0
            if y in val_Dic[time]:
                val = val_Dic[time][y]

            spamwriter.writerow([val])


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_data_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]

    # Make directory
    directory = os.path.dirname(src_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for raw_data_file in os.listdir(raw_data_dir):
        if raw_data_file.endswith(".txt"):
            # Get source data file
            src_data_file = src_data_dir + 'src_data_' + raw_data_file
            # Get target data file
            tar_data_file = tar_data_dir + 'tar_data_' + raw_data_file
            # Update raw_data_file
            raw_data_file = raw_data_dir + raw_data_file

            # Generate data
            generate_data()
