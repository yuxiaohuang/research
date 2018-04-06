"""
Please cite the following paper when using the code


"""


import sys
import os


def generate_setting(setting_file, result_dir):
    """
    Generate the setting
    :param setting_file: the pathname of the setting file
    :param result_dir: the pathname of the result directory
    :return:
    """

    # Make directory
    directory = os.path.dirname(setting_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Make directory
    directory = os.path.dirname(result_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the setting
    setting = """
###--------------------------------------------------------------------------------------------------------
### Parameter setting file for the classifiers
###--------------------------------------------------------------------------------------------------------

###--------------------------------------------------------------------------------------------------------
### The pathname of (or link to) the data file, must be specified
###--------------------------------------------------------------------------------------------------------

data_file = """ + data_dir + data_file + """

###--------------------------------------------------------------------------------------------------------
### The header, None by default
###--------------------------------------------------------------------------------------------------------

header = 

###--------------------------------------------------------------------------------------------------------
### The place holder for missing values, '?' by default
###--------------------------------------------------------------------------------------------------------

place_holder_for_missing_vals = 

###--------------------------------------------------------------------------------------------------------
### The (name of the) columns, must be specified
###--------------------------------------------------------------------------------------------------------

columns = """ + ' '.join(columns) + """

###--------------------------------------------------------------------------------------------------------
### The (name of the) target, must be specified
###--------------------------------------------------------------------------------------------------------

target = """ + target + """

###--------------------------------------------------------------------------------------------------------
### The (name of the) features that should be excluded, empty by default
###--------------------------------------------------------------------------------------------------------

exclude_features = 

###--------------------------------------------------------------------------------------------------------
### The (name of the) categorical features, empty by default
###--------------------------------------------------------------------------------------------------------

categorical_features = 

###--------------------------------------------------------------------------------------------------------
### The label encoder, LabelEncoder by default
###--------------------------------------------------------------------------------------------------------

encoder = 

###--------------------------------------------------------------------------------------------------------
### The percentage of the testing set, 0.3 by default
###--------------------------------------------------------------------------------------------------------

test_size = 

###--------------------------------------------------------------------------------------------------------
### The scaler, StandardScaler by default
###--------------------------------------------------------------------------------------------------------

scaler = 

###--------------------------------------------------------------------------------------------------------
### The random state, zero by default
###--------------------------------------------------------------------------------------------------------

random_state = 

###--------------------------------------------------------------------------------------------------------
### The maximum number of iterations, 100 by default
###--------------------------------------------------------------------------------------------------------

max_iter = 

###--------------------------------------------------------------------------------------------------------
### The value of C, 1 by default
###--------------------------------------------------------------------------------------------------------

C = 

###--------------------------------------------------------------------------------------------------------
### The pathname of the probability distribution figure directory, None by default
###--------------------------------------------------------------------------------------------------------

prob_dist_fig_dir = """ + result_dir + 'prob_dist_fig/' + """

###--------------------------------------------------------------------------------------------------------
### The name of the probability distribution figure, the name of the setting file by default
###--------------------------------------------------------------------------------------------------------

prob_dist_fig_name = 

###--------------------------------------------------------------------------------------------------------
### The type of the probability distribution figure, '.pdf' by default
###--------------------------------------------------------------------------------------------------------

prob_dist_fig_type = 

###--------------------------------------------------------------------------------------------------------
### The pathname of the probability distribution file directory, None by default
###--------------------------------------------------------------------------------------------------------

prob_dist_file_dir = """ + result_dir + 'prob_dist_file/' + """

###--------------------------------------------------------------------------------------------------------
### The name of the probability distribution file, the name of the setting file by default
###--------------------------------------------------------------------------------------------------------

prob_dist_file_name = 

###--------------------------------------------------------------------------------------------------------
### The type of the probability distribution file, '.csv' by default
###--------------------------------------------------------------------------------------------------------

prob_dist_file_type = 

###--------------------------------------------------------------------------------------------------------
### The pathname of the score file directory, None by default
###--------------------------------------------------------------------------------------------------------

score_file_dir = """ + result_dir + 'score_file/' + """

###--------------------------------------------------------------------------------------------------------
### The name of the score file, the name of the setting file by default
###--------------------------------------------------------------------------------------------------------

score_file_name = 

###--------------------------------------------------------------------------------------------------------
### The type of the score file, '.txt' by default
###--------------------------------------------------------------------------------------------------------

score_file_type = 

###--------------------------------------------------------------------------------------------------------
### The average for precision_recall_fscore_support, 'micro' by default
###--------------------------------------------------------------------------------------------------------

average = 

###--------------------------------------------------------------------------------------------------------
### The classifiers, by default:
    RandomForestClassifier,
    AdaBoostClassifier,
    MLPClassifier,
    KNeighborsClassifier,
    GaussianNB,
    DecisionTreeClassifier,
    LogisticRegression(multi_class='ovr'),
    LogisticRegression(multi_class='multinomial', solver='lbfgs'),
    LogisticRegression(multi_class='multinomial', solver='sag'),
    LogisticRegression(multi_class='multinomial', solver='newton-cg'),  
    GaussianProcessClassifier,
    SVC
###--------------------------------------------------------------------------------------------------------

classifiers = 

###--------------------------------------------------------------------------------------------------------
### The number of jobs to run in parallel, -1 by default (all CPUs are used)
###--------------------------------------------------------------------------------------------------------

n_jobs = 
"""

    # Write the setting into setting_file
    with open(setting_file, 'w') as f:
        f.write(setting + '\n')


if __name__=="__main__":
    # Get the pathname of the data directory from command line
    data_dir = sys.argv[1]

    # Get the pathname of the setting directory from command line
    setting_dir = sys.argv[2]

    # Get the pathname of the result directory from command line
    result_dir = sys.argv[3]

    # Get the (name of the) target
    target = sys.argv[4]

    # Get the (name of the) columns
    columns = sys.argv[5:]

    for data_file in os.listdir(data_dir):
        if data_file.endswith('.txt') or data_file.endswith('.csv'):
            # Get the pathname of the setting file
            setting_file = setting_dir + os.path.basename(data_file)

            # Generate the setting
            generate_setting(setting_file, result_dir)
