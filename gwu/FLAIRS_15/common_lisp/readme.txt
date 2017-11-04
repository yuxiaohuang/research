
**********************   Description   **********************
This is the instruction for using the lisp code in paper:
Huang, Y. X., and Kleinberg, S. Fast and Accurate Causal Inference from Time Series Data. FLAIRS. 2015.



The input of our method includes:
1. Continuous-valued time series data
2. Discretized time series data
3. A list of time windows (or lags), e.g.:
'((1 1) (2 2) (3 3))

The output includes relationships and their significance, e.g.:
X_U, Y_c, 1, 1, 5
meaning X = UP causes Y with time lag 1 and significance 5.



Specifically, our method includes 2 major steps
1. Preprocess data.
2. Calculate alpha (causal significance).
Details of the functions used in these steps can be seen in file "alpha.lisp"


Now let us get started!



**********************   Step 1: Preprocess data   **********************
;;1.1 start sbcl
sbcl --dynamic-space-size 1000

;;1.2 load files
(load "/path/to/code/split-sequence.lisp");;"split-sequence.lisp" can be seen at: http://www.cliki.net/SPLIT-SEQUENCE
(load "/path/to/code/csv-parser.lisp");;"csv-parser.lisp" can be seen at: https://github.com/sharplispers/csv-parser/blob/master/csv-parser.lisp
(load "/path/to/code/alpha.lisp")

;;1.3 define input and output files
(let ((discrete-time-series-file "/path/to/file/input-discrete.csv")
      (continuous-time-series-file "/path/to/file/input-continuous.csv")
      (alpha-file "/path/to/file/alpha.csv"))

;;1.4 get global variables
(get-global-variables (discrete-time-series-file continuous-time-series-file header transpose-p)

;;1.5 generate and test hypotheses
(loop for (r s) in window-list
      do (let ((hyp (generate-hypothese *alphabet* *alphabet-c* r s)))
	    (test-hypotheses hyp type)))



**********************   Step 2: Calculate alpha   **********************
;;calculate alpha for each relationship
(get-all-alpha alpha-file))



