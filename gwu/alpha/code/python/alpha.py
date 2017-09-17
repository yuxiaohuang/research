# This is the python code for discovering causal relationships from time series data
# Please cite the following paper when using the code
# Y. Huang and S. Kleinberg. Fast and Accurate Causal Inference from Time Series Data. Florida Artificial Intelligence Research Society Conference (FLAIRS), 2015.


# modules
from __future__ import division
from scipy import stats
import sys
import os
import csv
import numpy as np
import math


# Notations
# _L      : indicates the data structure is a list
# _LL     : indicates the data structure is a list of list
# _Dic    : indicates the data structure is a dictionary
# _L_Dic  : indicates the data structure is a dictionary, where the value is a list
# _LL_Dic : indicates the data structure is a dictionary, where the value is a list of list


# Define global variables
# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: [T(e | c_t)], which is a list of T(e | c_t). Here T(e | c_t) is the set of timepoints where e is measured in c_t's window (c_t is the instance of c occurring at time t)
T_e_ct_LL_Dic = {}

# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: T(e | c), which is the set of timepoints where e is measured in c's time window
T_e_c_L_Dic = {}

# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: T(c), which is the set of timepoints where c occurs 
T_c_L_Dic = {}

# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: N(e | c, x), which is the total number of e being measured in the intersection of c and x's windows
N_e_c_x_Dic = {}

# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: N(e | c), which is the total number of e being measured in c's window
N_e_c_Dic = {}

# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: E[e | c], which is the expectation of e conditioned on c
E_e_c_Dic = {}

# key: e, which is an effect
# val: E[e], which is the expectation of e
E_e_Dic = {}

# key: (c, r, s), which is a tuple including potential cause c, start and end of time window, r and s  
# val: true or false
#      true, if c is the potential cause of e measured at time t
#      false, otherwise
is_in_Xt_L_Dic = {}

# key: time t, which is the time of e being measured
# val: X_t, which is the set of potential causes of e measured at time t 
Xt_L_Dic = {}

# key: time t
# val: list of discrete or discretized vars occurring at t
disc_val_L_Dic = {}

# key: continuous_valued var
# val: list of var's continuous value at each time
cont_val_Dic = {}

# key: continuous_valued var
# val: list of time and var's continuous value at that time, i.e. [[time, val]]
time_cont_val_Dic = {}

# list of discrete or discretized vars in the time series
alphabet_disc = []

# list of continuous_valued vars in the time series
alphabet_cont = []

# key: e, which is the effect
# val: list of (c, r, s) where c is a potential cause, r and s the start and end of time window
relations_Dic = {}

# key: e->(c, r, s), which is first the effect then a tuple including potential cause c, start and end of time window, r and s
# val: alpha((c, r, s), e), which is the causal significance of c on e in time window [r, s]
alpha_Dic = {}

# calculate alpha (causal significance) for all relationships
# @param        alpha_file         file containing relationships and their causal significance
def get_all_alpha(alpha_file):
  # output heading
  with open(alpha_file, 'wb') as f:
    spamwriter = csv.writer(f, delimiter = ',')
    spamwriter.writerow(["cause", "effect", "window_start", "window_end", "alpha"])

  # calculate alpha for each relationship
  for e in relations_Dic:
    get_alpha(e, alpha_file)


# calculate alpha for one relationship
# @param        e                  a continuous_valued variable
# @param        alpha_file         file containing relationships and their causal significance (i.e. alpha)
def get_alpha(e, alpha_file):
  # the key is building a system of linear equations, A * alpha = B
  # where A is a matrix (denoted by A_LL) and B a column vector (denoted by B_L)
  # please see details of A and B in the FLAIRS paper
  # get the set of potential causes, X_L
  X_L = relations_Dic[e]
  if X_L:
    # get A_LL
    A_LL = get_A_LL(X_L)
    # get the rank of A_LL
    rank = np.linalg.matrix_rank(A_LL)
    # if A_LL is full rank (i.e. the system of linear equations has unique solutions)
    if rank == len(A_LL):
      # if A_LL is full rank
      # get B_L
      B_L = get_B_L(X_L, e)
      # solve the system of linear equations
      B_L = np.linalg.solve(A_LL, B_L)
      # output the relationships and their alpha (causal significance)
      get_alpha_file(X_L, e, B_L, alpha_file)
    else:
      # if A_LL is not full rank, get the subsystem of linear equations
      X_LIS_L = get_X_LIS_L(X_L, e)
      if X_LIS_L:
        # get A_LL 
        A_LIS_LL = get_A_LL(X_LIS_L)
        # get B_L
        B_LIS_array = get_B_L(X_LIS_L, e)
        # solve the subsystem of linear equations
        B_LIS_array = np.linalg.solve(A_LIS_LL, B_LIS_array)
        # output the relationships and their alpha (causal significance)
        get_alpha_file(X_LIS_L, e, B_LIS_array, alpha_file)


# get A_LL, the coefficient matrix of the system of linear equations
# @param        X_L                the set of potential causes
def get_A_LL(X_L):
  n = len(X_L)
  A_LL = []
  for i in range(n):
    row_L = []
    for j in range(n):
      c_T = X_L[i]
      x_T = X_L[j]
      N_e_c_x = get_N_e_c_x(c_T, x_T)
      N_e_x = get_N_e_c(x_T)
      T_e_c_L = get_T_e_c_L(c_T)
      N_e_c = get_N_e_c(c_T)
      len_T_e = len(disc_val_L_Dic)
      nominator = N_e_c_x * len_T_e - N_e_x * len(T_e_c_L)
      denominator = N_e_c * (len_T_e - len(T_e_c_L))
      f_e_c_x = nominator / denominator
      row_L.append(f_e_c_x)
    A_LL.append(row_L)
  return A_LL


# get N_e_c_x_Dic
# @param        c_T                (c, r, s), a tuple including potential cause c, start and end of time window, r and s
# @param        x_T                (x, r, s), a tuple including potential cause x, start and end of time window, r and s
def get_N_e_c_x(c_T, x_T):
  if not (c_T, x_T) in N_e_c_x_Dic:
    T_e_xt_LL = get_T_e_ct_LL(x_T)
    N_e_c_x = 0
    for T_e_xt_L in T_e_xt_LL:
      for time in T_e_xt_L:
        if is_in_Xt(c_T, time):
          N_e_c_x += 1
    N_e_c_x_Dic[(c_T, x_T)] = N_e_c_x
  return N_e_c_x_Dic[(c_T, x_T)]


# get T_e_ct_LL_Dic
# @param       c_T                 (c, r, s), a tuple including potential cause c, start and end of time window, r and s
def get_T_e_ct_LL(c_T):
  (c, r, s) = c_T
  if not c_T in T_e_ct_LL_Dic:
    T_c_L = get_T_c_L_Dic(c)
    T_e_ct_LL = []
    for tc in T_c_L:
      T_e_ct_L = []
      for time in range(tc + r, tc + s + 1):
        if time in disc_val_L_Dic:
          T_e_ct_L.append(time)
      if T_e_ct_L:
        T_e_ct_LL.append(T_e_ct_L)
    T_e_ct_LL_Dic[c_T] = T_e_ct_LL
  return T_e_ct_LL_Dic[c_T]


# get T_c_L_Dic
# @param        c                  a potential cause 
def get_T_c_L_Dic(c):
  if not c in T_c_L_Dic:
    T_c_L = []
    for time in disc_val_L_Dic:
      if c in disc_val_L_Dic[time]:
        T_c_L.append(time)
    T_c_L_Dic[c] = T_c_L
  return T_c_L_Dic[c]


# get T_e_c_L_Dic
# @param       c_T                 (c, r, s), a tuple including potential cause c, start and end of time window, r and s
def get_T_e_c_L(c_T):
  if not c_T in T_e_c_L_Dic:
    T_e_ct_LL = get_T_e_ct_LL(c_T)
    T_e_c_L = []
    for T_e_ct_L in T_e_ct_LL:
      for time in T_e_ct_L:
        T_e_c_L.append(time)
    T_e_c_L = list(set(T_e_c_L))
    T_e_c_L_Dic[c_T] = T_e_c_L
  return T_e_c_L_Dic[c_T]


# get N_e_c_Dic
# @param       c_T                 (c, r, s), a tuple including potential cause c, start and end of time window, r and s
def get_N_e_c(c_T):
  if not c_T in N_e_c_Dic:
    T_e_ct_LL = get_T_e_ct_LL(c_T)
    N_e_c = 0
    for T_e_ct_L in T_e_ct_LL:
      N_e_c += len(T_e_ct_L)
    N_e_c_Dic[c_T] = N_e_c
  return N_e_c_Dic[c_T]


# get B_L, the column vector of the system of linear equations
# @param        A_LL               the coefficient matrix of the system of linear equations
# @param        X_L                the set of potential causes
# @param        e                  an effect
def get_B_L(X_L, e):
  n = len(X_L)
  B_L = []
  for i in range(n):
    c_T = X_L[i]
    T_e_c_L = get_T_e_c_L(c_T)
    N_e_c = get_N_e_c(c_T)
    E_e_c = get_E_e_c(e, c_T)
    E_e = get_E_e(e)
    len_T_e = len(disc_val_L_Dic)
    nominator = len_T_e * len(T_e_c_L)
    denominator = N_e_c * (len_T_e - len(T_e_c_L))
    f_e_c = nominator / denominator
    B_L.append(f_e_c * (E_e_c - E_e))
  return B_L


# is c in Xt? i.e. is c a potential cause of e being measured at time t
# @param        c_T                (c, r, s), a tuple including potential cause c, start and end of time window, r and s
# @param        time               the time where e is measured
def is_in_Xt(c_T, time):
  if not c_T in is_in_Xt_L_Dic:
    T_e_c_L = get_T_e_c_L(c_T)
    for te in T_e_c_L:
      if not te in Xt_L_Dic:
        Xt_L_Dic[te] = []
      Xt_L_Dic[te].append(c_T)
    is_in_Xt_L_Dic[c_T] = True
  if time in Xt_L_Dic and c_T in Xt_L_Dic[time]:
    return True
  else:
    return False


# get alpha_file
# @param        X_L                the set of potential causes
# @param        e                  an effect
# @param        B_L                the column vector of the system of linear equations
# @param        alpha_file         file containing relationships and their causal significance
def get_alpha_file(X_L, e, B_L, alpha_file):
  with open(alpha_file, 'a') as f:
    for i in range(len(X_L)):
      (c, r, s) = X_L[i]
      alpha = B_L[i]
      spamwriter = csv.writer(f, delimiter = ',')
      spamwriter.writerow([c, e, r, s, alpha])
      # get alpha_Dic
      if not e in alpha_Dic:
        alpha_Dic[e] = {}
      alpha_Dic[e][(c, r, s)] = alpha


# get a linearly independent subset of X
# please see details of the method in the FLAIRS paper mentioned in the beginning
# @param        X_L                the set of potential causes
# @param        e                  an effect
def get_X_LIS_L(X_L, e):
  n = len(X_L)
  X_LIS_L = []
  c_T_abs_dif_L = []
  for c_T in X_L:
    c_T_abs_dif_L.append([c_T, abs(get_E_e_c(e, c_T) - get_E_e(e))])

  # bubble sort 
  for i in reversed(range(n - 1)):
    for j in range(i + 1):
      if c_T_abs_dif_L[j][1] < c_T_abs_dif_L[j + 1][1]:
        c_T_abs_dif_L[j], c_T_abs_dif_L[j + 1] = c_T_abs_dif_L[j + 1], c_T_abs_dif_L[j]

  # greedy search
  for [c_T, abs_dif] in c_T_abs_dif_L:
    X_LIS_L.append(c_T)
    if len(X_LIS_L) > 1:
      # get A_LIS_LL from X_LIS_L, which is the same as getting A_LL from X_L (in function get_alpha)
      A_LIS_LL = get_A_LL(X_LIS_L)
      # get the rank of A_LIS_LL 
      rank = np.linalg.matrix_rank(A_LIS_LL)
      # if A_LIS_LL is not full rank (i.e. the subsystem of linear equations has no unique solutions)
      if rank < len(A_LIS_LL):
        X_LIS_L.remove(c_T)
  return X_LIS_L


# get global_variables: disc_val_L_Dic, time_cont_val_Dic, cont_val_Dic, alphabet_disc, alphabet_cont
# @param        disc_data_file     discrete data file, which includes variables that can be the causes, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        cont_data_file     continuous data file, which includes variables that can be the effects, the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        header             tells us whether the data files have a header
#                                  True,  if there is a header
#                                  False, otherwise
# @param        transpose          tells us whether the data need to be transposed
#                                  False, when the data are of the above form
#                                  True,  when the data are of the following form
#                                  var1 (i.e. header), ..., varn (i.e. header)
#                                  var1_t1 (i.e. val), ..., varn_t1 (i.e. val)
#                                  , ...,
#                                  var1_tn (i.e. val), ..., varn_tn (i.e. val)
def get_global_variables(disc_data_file, cont_data_file, header, transpose):
  # get data from files
  disc_var_time_val_LLL = get_var_time_val_LLL(disc_data_file, header, transpose, "discrete")
  cont_var_time_val_LLL = get_var_time_val_LLL(cont_data_file, header, transpose, "continuous")

  # get disc_val_L_Dic
  # the key is time and the value is a list of discrete or discretized variables being true and the time
  # e.g. at time 1, "glucose" is "High" and "insulin" is "Low", then disc_val_L_Dic[1] = ["glucose_High", "insulin_Low"]
  for [var, time_val_LL] in disc_var_time_val_LLL:
    for [time, val] in time_val_LL:
      # discrete val is a string representing a category, e.g. "High", "Normal", or "Low"
      # check empty string
      if val:
        if not time in disc_val_L_Dic:
          disc_val_L_Dic[time] = []
        # concatenate the val to the name, e.g. var "glucose" with value "High" will be "glucose_High"
        disc_val_L_Dic[time].append(var + "_" + val)

  # get time_cont_val_Dic
  # the key is var (i.e. name) and the value is a list of [time, val]
  # e.g. "glucose" is 100 at time 1 and 110 at time 2, then cont_var_time_val_LLL["glucose"] = [[1, 100], [2, 110]]
  for [var, time_val_LL] in cont_var_time_val_LLL:
    for [time, val] in time_val_LL:
      # continuous val is a string representing a number, e.g. "100"
      # check string that is not a number
      if is_number(val):
        val = float(val)
        if not var in time_cont_val_Dic:
          time_cont_val_Dic[var] = []
        time_cont_val_Dic[var].append([time, val])

  # get cont_val_Dic
  # the first key is var (i.e. name) and the second key is time
  # the value is val
  # e.g. "glucose" is 100 at time 1, then cont_val_Dic["glucose"][1] = 100
  # this dictionary is used for cases where variables are not measured at every timepoint
  for [var, time_val_LL] in cont_var_time_val_LLL:
    for [time, val] in time_val_LL:
      # check string that is not a number
      if is_number(val):
        val = float(val)
        if not var in cont_val_Dic:
          cont_val_Dic[var] = {}
        cont_val_Dic[var][time] = val

  # get alphabet_disc
  # e.g., ["glucose_High", "insulin_low"]
  for time in disc_val_L_Dic:
    for var in disc_val_L_Dic[time]:
      if not var in alphabet_disc:
        alphabet_disc.append(var)

  # get alphabet_cont
  # e.g., ["glucose', 'insulin"]
  for var in time_cont_val_Dic:
    alphabet_cont.append(var)


# get [var, [time, val]]
# @param        data_file          the data are of the following form
#                                  var1 (i.e. header), var1_t1 (i.e. val), ..., var1_tn (i.e. val)
#                                  , ...,
#                                  varn (i.e. header), varn_t1 (i.e. val), ..., varn_tn (i.e. val)
# @param        header             tells us whether the data files have a header
#                                  True,  if there is a header
#                                  False, otherwise
# @param        transpose          tells us whether the data need to be transposed
#                                  False, when the data are of the above form, thus do not need to be transposed
#                                  True,  when the data are of the following form, thus need to be transposed
#                                  var1 (i.e. header), ..., varn (i.e. header)
#                                  var1_t1 (i.e. val), ..., varn_t1 (i.e. val)
#                                  , ...,
#                                  var1_tn (i.e. val), ..., varn_tn (i.e. val)
# @param        data_type          "discrete",   if discrete data
#                                  "continuous", if continuous_valued data
def get_var_time_val_LLL(data_file, header, transpose, data_type):
  with open(data_file, 'rb') as f:
    spamreader = list(csv.reader(f, delimiter = ','))
    if transpose:
      # transpose the data
      spamreader = zip(*spamreader)

    var_time_val_LLL = []
    for i in range(len(spamreader)):
      var = ""
      time_val_LL = []

      # get val_L
      val_L = []
      val_start = 0
      if header:
        # the name of the var lies in the first column in each row
        var = spamreader[i][0].strip()
        # the value of the var starts from the second column in each row
        val_start = 1
      else:
        # we use the number of the row as the name of the var in that row
        var = str(i)
      for j in range(val_start, len(spamreader[i])):
        val = spamreader[i][j].strip()
        val_L.append(val)

      # get time_val_LL
      for time in range(len(val_L)):
        time_val_LL.append([time, val_L[time]])

      # get var_time_val_LLL
      var_time_val_LLL.append([var, time_val_LL])
    return var_time_val_LLL


# check whether string is a number
# @param        val                a string
def is_number(val):
  try:
    float(val)
    return True
  except ValueError:
    return False


# generate hypotheses, where a hypothesis is of the following form
# [cause, effect, window_start, window_end], or [c, e, r, s] for simplicity
# @param        c_L                a list of discrete or discretized vars
# @param        e_L                a list of continuous_valued vars
# @param        r                  the start of a time window, i.e. r in window [r, s]
# @param        s                  the end of a time window, i.e. s in window [r, s]
def generate_hypotheses(c_L, e_L, r, s):
  c_e_r_s_L = []
  for e in e_L:
    for c in c_L:
      c_e_r_s_L.append([c, e, r, s])
  return c_e_r_s_L


# test hypotheses
# for hypothesis [c, e, r, s], 1) test whether c is a potential cause of e related to time window [r, s], and 2) get relations_Dic
# @param        hypotheses         a hypothesis is of form: [c, e, r, s]
# @param        rel_type           tells us the type of hypotheses we want to test
#                                  "not_equal" for hypotheses s.t. E[e | c] != E[e]
#                                  "positive"  for hypotheses s.t. E[e | c] > E[e]
#                                  "negative"  for hypotheses s.t. E[e_| c] < E[e]
#                                  "all"       for all hypotheses
def test_hypotheses(hypotheses, rel_type):
  for [c, e, r, s] in hypotheses:
    E_e_c = get_E_e_c(e, (c, r, s))
    E_e = get_E_e(e)
    # check whether E[e | c] and E[e] are valid
    if E_e_c != None and E_e != None:
      if rel_type == "not_equal":
        if E_e_c != E_e:
          add_relationship(c, e, r, s)
      elif rel_type == "positive":
        if E_e_c > E_e:
          add_relationship(c, e, r, s)
      elif rel_type == "negative":
        if E_e_c < E_e:
          add_relationship(c, e, r, s)
      elif rel_type == "all":
        add_relationship(c, e, r, s)


# get E[e|c]
# @param        e                  an effect
# @param        c_T                (c, r, s), a tuple including potential cause c, start and end of time window, r and s
def get_E_e_c(e, c_T):
  T_e_c_L = get_T_e_c_L(c_T)
  if T_e_c_L:
    if not (e, c_T) in E_e_c_Dic:
      val_L = []
      for time in T_e_c_L:
        if time in cont_val_Dic[e]:
          val_L.append(cont_val_Dic[e][time])
      # check whether val_L is empty
      if not val_L:
        return None
      E_e_c = np.mean(val_L)
      E_e_c_Dic[(e, c_T)] = E_e_c
    return E_e_c_Dic[(e, c_T)]
  else:
    return None


# get E[e]
# @param        e                  an effect
def get_E_e(e):
  if not e in E_e_Dic:
    val_L = []
    for time in cont_val_Dic[e]:
      val_L.append(cont_val_Dic[e][time])
    # check whether val_L is empty
    if not val_L:
      return None
    E_e = np.mean(val_L)
    E_e_Dic[e] = E_e
  return E_e_Dic[e]


# add relationship [c, e, r, s] to relations_Dic
# @param        c                  a potential cause
# @param        e                  an effect
# @param        r                  the start of a time window, i.e. r in window [r, s]
# @param        s                  the end of a time window, i.e. s in window [r, s]
def add_relationship(c, e, r, s):
  if not e in relations_Dic:
    relations_Dic[e] = []
  relations_Dic[e].append((c, r, s))


# get significant relationships based on significance test (z_test)
# @param        sig_rel_file       file containing significant relationships 
# @param        family_type        tells us whether to test the relationships for all effects (all families) at a time, or the relationships for each effect (one family) at a time
# @param        p_val cutoff       p value cutoff, which is used to decide significant relationships 
#                                  "one", significance test based on one family
#                                  "all", significance test based on all families
# @param        tail_type          tells us the type of z_test we want to do
#                                  "positive", if p_val < p_val_cutoff and z_val > 0
#                                  "negative", if p_val < p_val_cutoff and z_val < 0
#                                  "both",     if p_val < p_val_cutoff
def significance_test(sig_rel_file, family_type, p_val_cutoff, tail_type):
  # output heading
  with open(sig_rel_file, 'wb') as f:
    spamwriter = csv.writer(f, delimiter = ',')
    spamwriter.writerow(["cause", "effect", "window_start", "window_end", "alpha", "p_val"])

  if family_type == "all":
    # get sig_rel_file
    get_sig_rel_file(sig_rel_file, p_val_cutoff, tail_type, alpha_Dic)
  else:
    for e in alpha_Dic:
      # get sig_rel_file
      get_sig_rel_file(sig_rel_file, p_val_cutoff, tail_type, [e])


# get significant relationship file
# @param        sig_rel_file       file containing significant relationships 
# @param        p_val cutoff       p value cutoff, which is used to decide significant relationships 
# @param        tail_type          tells us the type of z_test we want to do
#                                  "positive", if p_val < p_val_cutoff and z_val > 0
#                                  "negative", if p_val < p_val_cutoff and z_val < 0
#                                  "both",     if p_val < p_val_cutoff
# @param        e_L                a list of effects
def get_sig_rel_file(sig_rel_file, p_val_cutoff, tail_type, e_L):
  # get alpha_LL
  alpha_LL = []
  for e in e_L:
    for (c, r, s) in alpha_Dic[e]:
      alpha = alpha_Dic[e][(c, r, s)]
      alpha_LL.append([c, e, r, s, alpha])

  # get alpha_L  
  alpha_L = [alpha for [c, e, r, s, alpha] in alpha_LL]

  # check if std == 0
  if np.std(alpha_L) == 0:
    return

  # get z_val_L
  z_val_L = stats.zscore(alpha_L)

  # get p_val_L
  p_val_L = stats.norm.sf([abs(z_val) for z_val in z_val_L])

  # get sig_rel_LL
  sig_rel_LL = []
  for i in range(len(alpha_LL)):
    if math.isnan(p_val_L[i]) or p_val_L[i] >= p_val_cutoff:
      continue
    if tail_type == "positive" and z_val_L[i] > 0 or tail_type == "negative" and z_val_L[i] < 0 or tail_type == "both":
      sig_rel_L = alpha_LL[i] + [p_val_L[i]]
      sig_rel_LL.append(sig_rel_L)

  # output sig_rel_L
  with open(sig_rel_file, 'a') as f:
    for sig_rel_L in sig_rel_LL:
      spamwriter = csv.writer(f, delimiter = ',')
      if sig_rel_L:
        spamwriter.writerow(sig_rel_L)


# main function
if __name__=="__main__":
  # get parameters
  # please see details of the parameters in the readme file
  disc_data_file = sys.argv[1]
  cont_data_file = sys.argv[2]
  header = sys.argv[3]
  transpose = sys.argv[4]
  rel_type = sys.argv[5]
  alpha_file = sys.argv[6]
  sig_rel_file = sys.argv[7]
  family_type = sys.argv[8]
  p_val_cutoff = float(sys.argv[9])
  tail_type = sys.argv[10]
  lag_L = sys.argv[11:]
  # get win_L from lag_L
  win_L = []
  for i in range(0, len(lag_L), 2):
    win = [int(lag_L[i]), int(lag_L[i + 1])]
    win_L.append(win)

  # make directory
  directory = os.path.dirname(alpha_file)
  if not os.path.exists(directory):
    os.makedirs(directory)
  directory = os.path.dirname(sig_rel_file)
  if not os.path.exists(directory):
    os.makedirs(directory)

  # get global variables
  get_global_variables(disc_data_file, cont_data_file, header, transpose)

  # generate and test hypotheses
  for [r, s] in win_L:
    hyp = generate_hypotheses(alphabet_disc, alphabet_cont, r, s)
    test_hypotheses(hyp, rel_type)

  # calculate alpha for all relationships
  get_all_alpha(alpha_file)

  # get significant relationships based on family_type,  p_val_cutoff, and tail_type
  significance_test(sig_rel_file, family_type, p_val_cutoff, tail_type)
