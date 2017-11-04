;;;This is the common lisp code for calculating causal significance using alpha

(use-package "SPLIT-SEQUENCE")

;;Define global variables

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s  
;;val: (list T(e | c_t)), i.e. a list of T(e | c_t), which is the set of timepoints where e is measured in c_t's window (c_t is the instance of c occurring at time t)
(defvar *T-e-ct-LL-hash* (make-hash-table :test #'equal))

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s
;;val: T(e | c), which is the set of timepoints where e is measured in c's time window
(defvar *T-e-c-L-hash* (make-hash-table :test #'equal))

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s
;;val: T(c), which is the set of timepoints where c occurs 
(defvar *T-c-L-hash* (make-hash-table :test #'equal))

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s
;;val: N(e | c, x), the total number of e being measured in the intersection of c and x's windows
(defvar *N-e-c-x-hash* (make-hash-table :test #'equal))

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s
;;val: N(e | c), the total number of e being measured in c's window
(defvar *N-e-c-hash* (make-hash-table :test #'equal))

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s
;;val: E[e | c], the expectation of e conditioned on c
(defvar *E-e-c-hash* (make-hash-table :test #'equal))

;;key: e, the effect
;;val: E[e], the expectation of e
(defvar *E-e-hash* (make-hash-table :test #'equal))

;;key: (list c r s), i.e. a list including potential cause c, start and end of time window, r and s
;;val: t (true) or nil (false)
;;t,   if c is the potential cause of e measured at time t
;;nil, otherwise
(defvar *Xt-L-p-hash* (make-hash-table :test #'equal))

;;key: t, the time of e being measured
;;val: X_t, the set of potential causes of e measured at time t 
(defvar *Xt-L-hash* (make-hash-table :test #'equal))

;;key: time t
;;val: list of discrete or discretized vars occurring at t
(defvar *discrete-value-hash* (make-hash-table :test #'equal))

;;key: continuous-valued var
;;val: list of var's continuous value at a time
(defvar *continuous-value-L-hash* (make-hash-table :test #'equal))

;;key: continuous-valued var
;;val: list of time and var's continuous value at a time, i.e. (list (list time val))
(defvar *time-continuous-value-L-hash* (make-hash-table :test #'equal))

;;list of discrete or discretized vars in the time series
(defvar *alphabet* (make-hash-table :test #'equal))

;;list of continuous-valued vars in the time series
(defvar *alphabet-c* (make-hash-table :test #'equal))

;;key: (list effect)
;;val: list of (c r s) where c is a potential cause, r and s the start and end of time window
(defvar *relations* (make-hash-table :test #'equal :synchronized t))


;;calculate alpha for all effects
;;@param        result-file         file containing relationships and their causal significance
(defmethod get-all-alpha (result-file)
  ;;output heading
  (with-open-file (out result-file :direction :output :if-exists :supersede)
		  (format out "cause, effect, start, end, alpha~%"))

  (loop for effect-L being the hash-keys of *relations*
	do (get-alpha effect-L result-file))
  )

;;calculate alpha for one effect
;;@param        effect-L            (list e), a list of e
;;@param        result-file         file containing relationships and their causal significance
(defmethod get-alpha (effect-L result-file)
  (let ((X-L (remove-duplicates (gethash effect-L *relations*) :test #'equal))
	(e (first effect-L)))       
    (if X-L
	(let ((A-array (get-A-array X-L))
	      (temp-array (get-A-array X-L)))
	  (if (is-full-rank temp-array)
	      (let ((B-array (get-B-array X-L e)))
		(setf B-array (solve-system-of-linear-equations A-array B-array))
		(get-result-file X-L e B-array result-file))
	    (let ((X-LIS-L (get-X-LIS-L X-L e)))
	      (if X-LIS-L
		  (let ((A-LIS-array (get-A-array X-LIS-L))
			(B-LIS-array (get-B-array X-LIS-L e)))
		    (setf B-LIS-array (solve-system-of-linear-equations A-LIS-array B-LIS-array))
		    (get-result-file X-LIS-L e B-LIS-array result-file))))))))
  )

;;get A-array, the coefficient matrix of the system of linear equations
;;@param        X-L                 the set of potential causes
(defmethod get-A-array (X-L)
  (format t "get-A-array~%")
  (let* ((n (length X-L))
	 (A-array (make-array (* n n))))
    (loop for i from 0 below n 
          do (loop for j from 0 below n 
                   do (let* ((c-L (nth i X-L))
                             (x-L (nth j X-L))
                             (N-e-c-x (get-N-e-c-x c-L x-L))
			     (N-e-x (get-N-e-c x-L))
			     (T-e-c-L (get-T-e-c-L c-L))
			     (N-e-c (get-N-e-c c-L))
		             (length-T-e (hash-table-count *discrete-value-hash*))
			     (nominator (- (* N-e-c-x length-T-e) (* N-e-x (length T-e-c-L))))
			     (denominator (* N-e-c (- length-T-e (length T-e-c-L))))
			     (f-e-c-x (if (= denominator 0) 
					  (format t "divided by 0 in get-A-array!~%")
					(/ nominator denominator))))
                        (setf (aref A-array (+ (* i n) j)) f-e-c-x))))
    A-array)			   
  )

;;get *N-e-c-x-hash*
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
;;@param        x-L                (list x r s), a list including potential cause x, start and end of time window, r and s
(defmethod get-N-e-c-x (c-L x-L)
  (if (gethash (list c-L x-L) *N-e-c-x-hash*)
      (gethash (list c-L x-L) *N-e-c-x-hash*)
    (let ((T-e-xt-LL (get-T-e-ct-LL x-L))
	  (N-e-c-x 0))
      (loop for T-e-xt-L in T-e-xt-LL
	    do (loop for time in T-e-xt-L
		     when (is-in-Xt c-L time)
		     do (incf N-e-c-x)))
      (setf (gethash (list c-L x-L) *N-e-c-x-hash*) N-e-c-x)))
  )

;;get *T-e-ct-LL-hash*
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
(defmethod get-T-e-ct-LL (c-L)
  (if (gethash c-L *T-e-ct-LL-hash*)
      (gethash c-L *T-e-ct-LL-hash*)
    (let* ((c (first c-L))
	   (r (second c-L))
	   (s (third c-L))
	   (T-c-L (get-T-c-L c))
	   (T-e-ct-LL nil))
      (loop for tc in T-c-L
	    do (let ((T-e-ct-L (loop for time from (+ tc r) to (+ tc s)
				     when (gethash time *discrete-value-hash*) ;;when time is in the time series
				     collecting time)))					
		 (if T-e-ct-L
		     (push T-e-ct-L T-e-ct-LL))))
      (setf (gethash c-L *T-e-ct-LL-hash*) T-e-ct-LL)))
  )

;;get *T-c-L-hash*
;;@param        c                   a potential cause 
(defmethod get-T-c-L (c)
  (if (gethash c *T-c-L-hash*)
      (gethash c *T-c-L-hash*)
    (setf (gethash c *T-c-L-hash*)
	  (loop for time being the hash-keys of *discrete-value-hash*
		when (find c (gethash time *discrete-value-hash*) :test #'equal)
		collecting time)))
  )

;;get *T-e-c-L-hash*
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
(defmethod get-T-e-c-L (c-L)
  (if (gethash c-L *T-e-c-L-hash*)
      (gethash c-L *T-e-c-L-hash*)
    (let ((T-e-ct-LL (get-T-e-ct-LL c-L))
	  (T-e-c-L nil))
      (loop for T-e-ct-L in T-e-ct-LL
	    do (loop for time in T-e-ct-L
		     do (push time T-e-c-L)))
      (setf T-e-c-L (remove-duplicates T-e-c-L :test #'equal))
      (setf (gethash c-L *T-e-c-L-hash*) T-e-c-L)))
  )

;;get *N-e-c-hash*
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
(defmethod get-N-e-c (c-L)
  (if (gethash c-L *N-e-c-hash*)
      (gethash c-L *N-e-c-hash*)
    (let ((T-e-ct-LL (get-T-e-ct-LL c-L))
	  (N-e-c 0))
      (loop for T-e-ct-L in T-e-ct-LL
	    do (incf N-e-c (length T-e-ct-L)))
      (setf (gethash c-L *N-e-c-hash*) N-e-c)))
  )

;;is A-array full rank?
;;@param        A-array             the coefficient matrix of the system of linear equations
(defmethod is-full-rank (A-array)
  (format t "is-full-rank~%")
  (let ((n (isqrt (length A-array)))
	(full-rank-p t))
    ;;gauss elimination
    (loop for k from 0 to (- n 2)
          ;;find the max row and switch it with row k
          do (let ((max (abs (aref A-array (+ (* k n) k))))
                   (max-row k))
					;find the max row
               (loop for i from (+ k 1) below n
                     when (> (abs (aref A-array (+ (* i n) k))) max)
                     do (progn
			  (setf max (abs (aref A-array (+ (* i n) k))))
			  (setf max-row i)))
               (when (= max 0)
                 (setf full-rank-p nil)
                 (return))
               ;;switch it with row k
               (if (/= k max-row)
		   (loop for j from k below n
			 do (rotatef (aref A-array (+ (* max-row n) j))
				     (aref A-array (+ (* k n) j))))))
	  ;;division
	  do (let ((denominator (aref A-array (+ (* k n) k))))
               (loop for j from k below n
                     do (setf (aref A-array (+ (* k n) j))
                              (/ (aref A-array (+ (* k n) j)) denominator))))
	  ;;subtraction
	  do (loop for i from (+ k 1) below n
                   do (let ((multiplier (aref A-array (+ (* i n) k))))
                        (loop for j from k below n
                              do (decf (aref A-array (+ (* i n) j))
                                       (* (aref A-array (+ (* k n) j)) multiplier))))))
    ;;return 
    (if (or (not full-rank-p)
            (= (aref A-array (+ (* (- n 1) n) (- n 1))) 0))
	nil
      t))
  )

;;check whether A is full rank in a greedy fashion
;;@param        A-array             the coefficient matrix of the system of linear equations
(defmethod is-full-rank-greedy (A-array)
  (format t "is-full-rank-greedy~%")
  (let* ((N (isqrt (length A-array)))
	 (N-local (- N 2))
	 (full-rank-p t))

    ;;gaussian elimination
    (loop for k from 0 to (- N 2)
	  ;;update the denominator
	  do (if (and (> N-local 0)
		      (= k N-local))
		 (setf (aref A-array (+ (* N-local N) N-local))
		       (gethash (list N-local N-local) *coefficient-hash*)))

          ;;find the max row and switch it with row k
          do (let ((max-row k))
	       (if (>= k N-local)
		   (let ((max (abs (aref A-array (+ (* k N) k)))))
		     ;;find the max row
		     (loop for i from (+ k 1) below N
			   when (> (abs (aref A-array (+ (* i N) k))) max)
			   do (progn
				(setf max (abs (aref A-array (+ (* i N) k))))
				(setf max-row i)))

		     ;;check rank
		     (when (= max 0)
		       (setf full-rank-p nil)
		       (return))

		     ;;record max row
		     (setf (gethash k *max-row-hash*) max-row))
		 ;;restore max-row
		 (setf max-row (gethash k *max-row-hash*)))

	       ;;switch it with row k
	       (if (/= k max-row)
		   (loop for j from k below N
			 do (rotatef (aref A-array (+ (* max-row N) j))
				     (aref A-array (+ (* k N) j))))))

          ;;division
	  do (let ((denominator (if (< k N-local)
				    (gethash k *denominator-hash*)
				  (setf (gethash k *denominator-hash*)
					(aref A-array (+ (* k N) k))))))
               (loop for j from k below N
		     when (or (> j N-local)
			      (= j k N-local))
                     do (setf (aref A-array (+ (* k N) j))
                              (/ (aref A-array (+ (* k N) j)) denominator))))   

	  ;;subtraction
	  do (loop for i from (+ k 1) below N
                   do (let ((multiplier (if (<= i N-local)
					    (gethash (list i k) *multiplier-hash*)
					  (setf (gethash (list i k) *multiplier-hash*)
						(aref A-array (+ (* i N) k))))))
                        (loop for j from k below N
			      when (or (> i N-local)
				       (> j N-local))
                              do (decf (aref A-array (+ (* i N) j))
				       (* (if (and (<= j N-local)
						   (< k N-local))
					      (gethash (list k j) *coefficient-hash*)
					    (aref A-array (+ (* k N) j))) 
					  multiplier))))))
    
    ;;return 
    (if (or (not full-rank-p)
            (= (aref A-array (+ (* (- N 1) N) (- N 1))) 0))
	nil
      (progn
	;;update *coefficient-hash*
	(loop for i from 0 below N
	      do (loop for j from 0 below N
		       when (or (> i N-local)
				(> j N-local)
				(= i j N-local))
		       do (setf (gethash (list i j) *coefficient-hash*)
				(aref A-array (+ (* i N) j)))))
	t)))
  )

;;get B-array, the value vector of the system of linear equations
;;@param        A-array             the coefficient matrix of the system of linear equations
;;@param        X-L                 the set of potential causes
;;@param        e                   the effect
(defmethod get-B-array (X-L e)
  (let* ((n (length X-L))
	 (B-array (make-array n)))
    (loop for i from 0 below n 
          do (let* ((c-L (nth i X-L))
		    (T-e-c-L (get-T-e-c-L c-L))
	            (N-e-c (get-N-e-c c-L))
		    (E-e-c (get-E-e-c e c-L))
		    (E-e (get-E-e e))
		    (length-T-e (hash-table-count *discrete-value-hash*))
	       	    (nominator (* length-T-e (length T-e-c-L)))
		    (denominator (* N-e-c (- length-T-e (length T-e-c-L))))
		    (f-e-c (if (= denominator 0)
			       (format t "divided by 0 in get-B-array!~%")
			     (/ nominator denominator))))
	       (setf (aref B-array i) (* f-e-c (- E-e-c E-e)))))
    B-array)
  )

;;is c in Xt? i.e. is c a potential cause of e being measured at time t
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
;;@param        time               the time where e is measured
(defmethod is-in-Xt (c-L time)
  (if (not (gethash c-L *Xt-L-p-hash*))
      (let ((T-e-c-L (get-T-e-c-L c-L)))
	(loop for te in T-e-c-L
	      do (push c-L (gethash te *Xt-L-hash*)))
	(setf (gethash c-L *Xt-L-p-hash*) t)))
  (if (find c-L (gethash time *Xt-L-hash*) :test #'equal)
      t
    nil)
  )

;;solve system of linear equations
;;@param        A-array             the coefficient matrix of the system of linear equations
;;@param        A-array             the value vector of the system of linear equations
(defmethod solve-system-of-linear-equations (A-array B-array)
  (format t "solve-system-of-linear-equations~%")
  (let* ((n (isqrt (length A-array))))
    ;;gauss elimination
    (loop for k from 0 to (- n 2)
          ;;find the max row and switch it with row k
          do (let ((max (abs (aref A-array (+ (* k n) k))))
                   (max-row k))
	       ;;find the max row
               (loop for i from (+ k 1) below n
                     when (> (abs (aref A-array (+ (* i n) k))) max)
                     do (progn
			  (setf max (abs (aref A-array (+ (* i n) k))))
			  (setf max-row i)))
               ;;switch it with row k
               (when (/= k max-row)
                 (loop for j from k below n
                       do (rotatef (aref A-array (+ (* max-row n) j))
                                   (aref A-array (+ (* k n) j))))
		 (rotatef (aref B-array max-row) (aref B-array k))))
	  ;;division
	  do (let ((denominator (aref A-array (+ (* k n) k))))
               (loop for j from k below n
                     do (setf (aref A-array (+ (* k n) j))
                              (/ (aref A-array (+ (* k n) j)) denominator)))
	       (setf (aref B-array k) (/ (aref B-array k) denominator)))
	  ;;subtraction
	  do (loop for i from (+ k 1) below n
                   do (let ((multiplier (aref A-array (+ (* i n) k))))
                        (loop for j from k below n
                              do (decf (aref A-array (+ (* i n) j))
                                       (* (aref A-array (+ (* k n) j)) multiplier)))
			(decf (aref B-array i)
                              (* (aref B-array k) multiplier)))))
    ;;get solution
    (setf (aref B-array (- n 1))
          (/ (aref B-array (- n 1))
             (aref A-array (+ (* (- n 1) n) (- n 1)))))
    (loop for i from (- n 2) downto 0
          do (loop for j from (+ i 1) below n
                   do (decf (aref B-array i)
                            (* (aref A-array (+ (* i n) j)) (aref B-array j)))))
    B-array)
  )

;;get result-file
;;@param        X-L                 the set of potential causes
;;@param        e                   the effect
;;@param        B-array             the value vector of the system of linear equations
;;@param        result-file         file containing relationships and their causal significance
(defmethod get-result-file (X-L e B-array result-file)
  (loop for i from 0 below (length X-L)
	do (let ((c-L (nth i X-L)))
	     (with-open-file (out result-file :direction :output :if-exists :append)
			     (format out "~a, ~a, ~a, ~a, ~f~%" (first c-L) e (second c-L) (third c-L) (aref B-array i)))))
  )

;;get a linearly independent subset of X
;;@param        X-L                 the set of potential causes
;;@param        e                   the effect
(defmethod get-X-LIS-L (X-L e)
  (format t "get-X-LIS-L~%")
  (let ((n (length X-L))
	(X-LIS-L nil)
	(c-L-abs-dif-L (loop for c-L in X-L
			     collecting (list c-L (abs (- (get-E-e-c e c-L)
							  (get-E-e e)))))))
    (loop for i from (- n 2) downto 0 
	  do (loop for j from 0 to i
		   when (< (second (nth j c-L-abs-dif-L))
			   (second (nth (+ j 1) c-L-abs-dif-L)))
		   do (progn
			(rotatef (first (nth j c-L-abs-dif-L))
				 (first (nth (+ j 1) c-L-abs-dif-L)))
		       	(rotatef (second (nth j c-L-abs-dif-L))
				 (second (nth (+ j 1) c-L-abs-dif-L))))))

    (setf *max-row-hash* (make-hash-table :test #'equal)) ;;key: k, value: max-row
    (setf *coefficient-hash* (make-hash-table :test #'equal)) ;;key: '(i j), value: coefficient
    (setf *denominator-hash* (make-hash-table :test #'equal)) ;;key: k, value: denominator
    (setf *multiplier-hash* (make-hash-table :test #'equal)) ;;key: '(i j), value: multiplier
    
    ;;greedy search
    (loop for (c-L abs-dif) in c-L-abs-dif-L
	  do (setf X-LIS-L (append X-LIS-L (list c-L)))
          do (if (> (length X-LIS-L) 1)
	  ;;do (if (> (length c-L) 1)
		 (let ((A-LIS-array (get-A-array X-LIS-L)))
		   (if (not (is-full-rank-greedy A-LIS-array))
		       (setf X-LIS-L (remove c-L X-LIS-L :test #'equal))))))
    X-LIS-L)
  )

;;parse decimal from string
;;this function is built on the function "parse-decimal" which can be seen at http://www.cliki.net/DECIMAL-NUMBER
;;@param        str                 a string
(defun parse-decimal (str)
  (let* ((par (split-sequence #\E str))
         (parts (split-sequence #\. (first par)))
         (a_intermediate (car parts))
         (a (if a_intermediate
		(if (equal a_intermediate "")
		    "0"
		  a_intermediate)))
         (b (cadr parts))
         (a1 (parse-integer a))
         (b1 (or (parse-integer (or b "0") :junk-allowed t) 0))
         (mult (if (= (length par) 2)
		   (expt 10 (parse-decimal (second par)))
                 1)))
    (* (+ a1 (/ b1 (expt 10 (length b)))) mult)))


;;parse decimal from string using E notation
;;@param        str                 a string
(defun parse-decimal-w-exp (str) 
  (let* ((parts-w-exp  (if (= (length (split-sequence #\E str :test #'equal)) 2)
                           (split-sequence #\E str :test #'equal)
			 (split-sequence #\e str :test #'equal)))
         (parts-w-neg (split-sequence #\- (first parts-w-exp)))
         (parts (split-sequence #\. (first (last parts-w-neg))))
         (a (car parts)) 
         (b (cadr parts)) 
         (a1 (if (equal a "")
                 0
	       (parse-integer a)))
         (b1 (or (parse-integer (or b "0") :junk-allowed t) 0))
         (to-return (+ a1 (/ b1 (expt 10 (length b))))))
    (if (= 2 (length parts-w-exp))
        (setf to-return (* to-return (expt 10 (parse-decimal (second parts-w-exp))))))
    (if (= (length parts-w-neg) 2)
        (* -1 to-return)
      to-return))
  )

;;get global-variables: *discrete-value-hash*, *time-continuous-value-L-hash*, *alphabet*, *alphabet-c*
;;@param        discrete-time-series-file     time series data of form
;;                                            var1, var1_t1, var1_t2, ..., var1_tn
;;                                            var2, var2_t1, var2_t2, ..., varn_tn
;;@param        continuous-time-series-file   time series data of form
;;                                            var1, var1_t1, var1_t2, ..., var1_tn
;;                                            var2, var2_t1, var2_t2, ..., varn_tn
;;@param        header                        t,   if there is header
;;                                            nil, otherwise
;;@param        transpose-p                   tells us whether the data need to be transposed
;;                                            no when the data are of the above form
;;                                            yes hen the data are of the following form
;;                                            var1,      var2   , ..., varn
;;                                            var1_t1,   var2_t1, ..., varn_t1
(defmethod get-global-variables (discrete-time-series-file continuous-time-series-file header transpose-p)
  (format t "~%discrete-file: ~a~%" (pathname-name discrete-time-series-file))
  (format t "continuous-file: ~a~%~%" (pathname-name continuous-time-series-file))

  (let* ((discrete-var-time-value-LLL (get-var-time-value-LLL discrete-time-series-file header transpose-p "discrete"))
	 (continuous-var-time-value-LLL (get-var-time-value-LLL continuous-time-series-file header transpose-p "continuous")))

    ;;initialize global variables
    (clrhash *relations*)
    (clrhash *discrete-value-hash*)
    (clrhash *time-continuous-value-L-hash*)
    (clrhash *continuous-value-L-hash*)

    ;;get *discrete-value-hash*
    (loop for (var time-value-LL) in discrete-var-time-value-LLL
          do (loop for (time value) in time-value-LL
		   when value;;ignore nil
		   do (push value (gethash time *discrete-value-hash*))))

    ;;get *time-continuous-value-L-hash*
    (loop for (var time-value-LL) in continuous-var-time-value-LLL
          do (loop for (time value) in time-value-LL
		   when (and time value);;ignore nil
		   do (push (list time value) (gethash var *time-continuous-value-L-hash*))))

    ;;get *continuous-value-L-hash*
    ;;for cases where variables are not measured at every timepoint
    (loop for (var time-value-LL) in continuous-var-time-value-LLL
          do (loop for (time value) in time-value-LL
		   when (and time value);;ignore nil
		   do (progn
			(if (not (gethash var *continuous-value-L-hash*))
			    (setf (gethash var *continuous-value-L-hash*)
				  (make-hash-table :test #'equal)))
			(setf (gethash time (gethash var *continuous-value-L-hash*)) value))))

    ;;get *alphabet*
    (loop for h being the hash-keys of *discrete-value-hash*
          appending (gethash h *discrete-value-hash*) into temp
          finally (setf *alphabet* (remove-duplicates temp :test #'equal)))

    ;;get *alphabet-c*
    (setf *alphabet-c* (loop for h being the hash-keys of *time-continuous-value-L-hash*
                             collecting h)))
  )		  

;;get (list var (list (list time value)
;;@param        time-series-file   time series data of form
;;                                 var1, var1_t1, var1_t2, ..., var1_tn
;;                                 var2, var2_t1, var2_t2, ..., varn_tn
;;@param        header             t,   if there is header
;;                                 nil, otherwise
;;@param        transpose-p        tells us whether the data need to be transposed
;;                                 no when the data are of the above form
;;                                 yes hen the data are of the following form
;;                                 var1,      var2   , ..., varn
;;                                 var1_t1,   var2_t1, ..., varn_t1
;;@param        data-type          "discrete",   if discrete data
;;                                 "continuous", if continuous-valued data
(defmethod get-var-time-value-LLL (time-series-file header transpose-p data-type)
  (let* ((lines (if transpose-p
		    (transpose (csv-lines time-series-file))
		  (csv-lines time-series-file)))
	 (var-time-value-LLL (loop for i from 0 to (- (length lines) 1)
				   collecting (list (if header 
							(first (nth i lines))
						      (write-to-string i))
						    (loop with j = -1 
							  for value in (if header
									   (rest (nth i lines))
									 (nth i lines))
							  do (incf j)
							  collecting (list j (if (not (equal value "NIL"))
										 (if (equal data-type "discrete")
										     value			       						  
										   (parse-decimal-w-exp value))
									       nil)))))))
    var-time-value-LLL)
  )

;;generate hypotheses for an effect
;;a hypothesis is of form: (cause effect window-start window-end), or (c e r s) for simplicity
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
;;@param        e-L                (list effect)
;;@param        r                  the start of a time window, i.e. r in window [r, s]
;;@param        s                  the end of a time window, i.e. s in window [r, s]
(defmethod generate-hypotheses (c-L e-L r s)
  (loop for e in e-L
        appending (loop for c in c-L
			collecting (list c e r s)))
  )

;;;test hypotheses
;;for hypothesis (c e r s), test whether c is a potential cause of e related to time window [r, s] and get *relations*
;;@param        hypotheses          a hypothesis is of form: (c e r s)
;;@param        relationship-type   type of relationships we want to test
(defmethod test-hypotheses (hypotheses relationship-type)
  (loop for (c e r s) in hypotheses
	when (and c e r s)
        do (let ((E-e-c (get-E-e-c e (list c r s)))
		 (E-e (get-E-e e)))
	     (if E-e-c
		 (cond ((equal relationship-type "not-equal")
			(if (/= E-e-c E-e)
			    (add-relationship c e r s)))
		       ((equal relationship-type "positive")
			(if (> E-e-c E-e)
			    (add-relationship c e r s)))
		       ((equal relationship-type "negative")
			(if (< E-e-c E-e)
			    (add-relationship c e r s)))
		       ((equal relationship-type "all")
			(add-relationship c e r s))))))
  )

;;;get E[e|c]
;;@param        e                   the effect
;;@param        c-L                (list c r s), a list including potential cause c, start and end of time window, r and s
(defmethod get-E-e-c (e c-L)
  (let ((T-e-c-L (get-T-e-c-L c-L)))
    (if T-e-c-L
	(if (gethash (list e c-L) *E-e-c-hash*)
	    (gethash (list e c-L) *E-e-c-hash*)
	  (let* ((value-L (loop for time in T-e-c-L
				collecting (gethash time (gethash e *continuous-value-L-hash*))))
		 (E-e-c (get-mean value-L)))
	    (setf (gethash (list e c-L) *E-e-c-hash*) E-e-c)))
      nil))
  )

;;;get E[e]
;;@param        e                   the effect
(defmethod get-E-e (e)
  (if (gethash e *E-e-hash*)
      (gethash e *E-e-hash*)
    (let* ((value-L (loop for time being the hash-keys of (gethash e *continuous-value-L-hash*)
			  collecting (gethash time (gethash e *continuous-value-L-hash*))))
	   (E-e (get-mean value-L)))
      (setf (gethash e *E-e-hash*) E-e)))
  )

;;;add relationship (c e r s) to *relations*
;;@param        c                  a potential cause
;;@param        e                  an effect
;;@param        r                  the start of a time window, i.e. r in window [r, s]
;;@param        s                  the end of a time window, i.e. s in window [r, s]
(defmethod add-relationship (c e r s)
  (push (list c r s)
        (gethash (list e) *relations*))
  )

;;transpose the lines of a file
;;@param        x                   a list of lines of a file
(defmethod transpose (x)
  (let ((width (length (first x)))
        (height (length x)))

    (loop for i from 0 to (- width 1)
          collecting(loop for j from 0 to (- height 1)
                          collecting(nth i (nth j x))))
    
    ))

;;get the lines of a csv file
;;@param        in-file             the input csv file
(defmethod csv-lines (in-file)
  (with-open-file (stream in-file :direction :input)
					;(with-open-file (out out-file :direction :output)
		  (loop as line = (csv-parser::read-csv-line stream)
			while line
			collecting line)))
					;do (format out "~{~s~^,~}~%"  line)))))

;;;get mean value from value-L
;;@param        value-L             a list of value
(defmethod get-mean (value-L)
  (if (= (length value-L) 0)
      0
    (/ (reduce '+ value-L)
       (length value-L)))
  )


