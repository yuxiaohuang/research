
###-----------------------------------------------------------------------------------------------------------------------------
This is the instruction for running the code for our paper accepted to FLAIRS 2020:
"Improving Classification Accuracy by Mining Deterministic and Frequent Rules"


1. Place folder "arc" in your working directory

2. Skip this step if you are fine with running the code in parallel using 10 CPU cores
   Otherwise, use the following commands to specify n_jobs (number of CPU cores used when parallelizing, the default value is 10):
   
   vim ./arc/code/script/run_all.txt
   vim ./arc/code/python/ours/Setting.py
   vim ./arc/code/python/others/Setting.py

3. Use the following command to run the code:

   ./arc/code/script/run_all.txt

###-----------------------------------------------------------------------------------------------------------------------------



###-----------------------------------------------------------------------------------------------------------------------------
These are the requirements for the environments. Please note that they are not the minimum requirements. Instead, they are the ones that enable the above instruction to work.


1. Install Anaconda
   Go to: https://docs.anaconda.com/anaconda/install/ (or google install anaconda when broken link)
   Go to the "System requirements" section, choose your operating system and follow the corresponding instructions
   
2. Install joblib (for running the code in parallel)
   Use the following command to install joblib:

   conda install joblib

3. Install imblearn (for oversampling the minority classes)
   Use the following command to install imblearn:

   conda install -c conda-forge imbalanced-learn

4. Install graphviz (for generating the dot and pdf file)
   Use the following command to install graphviz:

   conda install -c anaconda graphviz 

###-----------------------------------------------------------------------------------------------------------------------------


