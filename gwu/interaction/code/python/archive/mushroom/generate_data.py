

# Please cite the following paper when using the code


# Modules
import sys
import os
import csv
import numpy as np
import math
import random


# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Global variables
# The list of name of variables
name_L = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# The list of full value of variables
val_ful_LL = [['edible=e,poisonous=p'],
               ['bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s'],
               ['fibrous=f,grooves=g,scaly=y,smooth=s'],
               ['brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y'],
               ['bruises=t,no=f'],
               ['almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s'],
               ['attached=a,descending=d,free=f,notched=n'],
               ['close=c,crowded=w,distant=d'],
               ['broad=b,narrow=n'],
               ['black=k,brown=n,buff=b,chocolate=h,gray=g,green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y'],
               ['enlarging=e,tapering=t'],
               ['bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?'],
               ['fibrous=f,scaly=y,silky=k,smooth=s'],
               ['fibrous=f,scaly=y,silky=k,smooth=s'],
               ['brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y'],
               ['brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y'],
               ['partial=p,universal=u'],
               ['brown=n,orange=o,white=w,yellow=y'],
               ['none=n,one=o,two=t'],
               ['cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z'],
               ['black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y'],
               ['abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y'],
               ['grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d']]


# The dictionary of source value
# key: time->var
# val: value of var at the time
src_val_Dic = {}

# The dictionary of target value
# key: time->var
# val: value of var at the time
tar_val_Dic = {}


# Generate source and target data
def generate_data():
    # Get val_acr_LL
    val_acr_LL = get_val_acr_LL()

    # Load the raw file
    with open(raw_file, 'r') as f:
        spamreader = list(csv.reader(f, delimiter = ','))

        # Get the maximum time stamp
        max_time_stamp = len(spamreader)

        # From the first line to the last (since there is no header)
        for i in range(0, max_time_stamp):
            for j in range(0, len(name_L)):
                # Get the name of the var
                name = name_L[j]

                # Get the list of val acronym of the var
                val_acr_L = val_acr_LL[j]

                # Get the value of the var
                val = spamreader[i][j].strip()

                if name == 'class':
                    # Update tar_val_Dic
                    get_val_Dic(name, val, val_acr_L, i + 1, tar_val_Dic)
                else:
                    # Update src_val_Dic
                    get_val_Dic(name, val, val_acr_L, i, src_val_Dic)

    # Write the source file
    with open(src_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Get the header
        src_L = []
        for source in src_val_Dic[0]:
            src_L.append(source)

        # Write the header
        spamwriter.writerow(src_L)

        # Write the value
        for time in range(0, max_time_stamp):
            # The list of value at the time
            val_L = []

            if time in src_val_Dic:
                for source in src_L:
                    if source in src_val_Dic[time]:
                        val_L.append(src_val_Dic[time][source])
                    else:
                        val_L.append(-1)
            else:
                for source in src_L:
                    val_L.append(-1)

            spamwriter.writerow(val_L)

    # Write the target file
    with open(tar_data_file, 'w') as f:
        spamwriter = csv.writer(f, delimiter=',')

        # Get the header
        tar_L = []
        for target in tar_val_Dic[1]:
            tar_L.append(target)

        # Write the header
        spamwriter.writerow(tar_L)

        # Write the value
        for time in range(0, max_time_stamp + 1):
            # The list of value at the time
            val_L = []

            if time in tar_val_Dic:
                for target in tar_L:
                    if target in tar_val_Dic[time]:
                        val_L.append(tar_val_Dic[time][target])
                    else:
                        val_L.append(-1)
            else:
                for target in tar_L:
                    val_L.append(-1)

            spamwriter.writerow(val_L)


# Get val_acr_LL
def get_val_acr_LL():
    # The list of val in acronym, empty by default
    val_acr_LL = []

    for val_ful_L in val_ful_LL:
        val_ful_spl_L = val_ful_L[0].split(',')
        val_acr_L = []

        for val_ful_spl in val_ful_spl_L:
            # Get the acronym, which is the last character of the string
            val_acr = val_ful_spl[-1]
            val_acr_L.append(val_acr)

        val_acr_LL.append(val_acr_L)

    return val_acr_LL


# Update val_Dic
def get_val_Dic(name, val, val_acr_L, time, val_Dic):
    # Initialization
    if not time in val_Dic:
        val_Dic[time] = {}

    # Get name_val
    name_val = name + '_' + val

    for val_acr in val_acr_L:
        # Get name_val_acr
        name_val_acr = name + '_' + val_acr
        if name_val_acr == name_val:
            val_Dic[time][name_val_acr] = 1
        else:
            val_Dic[time][name_val_acr] = 0


# Main function
if __name__=="__main__":
    # get parameters from command line
    # please see details of the parameters in the readme file
    raw_file_dir = sys.argv[1]
    src_data_dir = sys.argv[2]
    tar_data_dir = sys.argv[3]

    # Make directory
    directory = os.path.dirname(src_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = os.path.dirname(tar_data_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for raw_file in os.listdir(raw_file_dir):
        if raw_file.endswith(".txt"):
            # Get src data file
            src_data_file = src_data_dir + 'src_data' + '_' + raw_file
            # Get tar data file
            tar_data_file = tar_data_dir + 'tar_data' + '_' + raw_file

            # Update raw_file
            raw_file = raw_file_dir + raw_file

            # Generate data
            generate_data()
