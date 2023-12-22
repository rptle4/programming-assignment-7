from helpers import gram_schmidt
from structures import Vec, Matrix
import numpy as np
import cmath


# ----------------------- PROBLEM 1 ----------------------- #
def qr_solve(A: Matrix, b: Vec):
  """
    Solves the system of equations Ax = b by using the
    QR factorization of Matrix A
    :param A: Matrix of coefficients of the system
    :param b: Vec of constants
    :return:  Vec solution to the system
    """
  # Constructing U
  # U should be the set of orthonormal vectors returned
  # by applying Gram-Schmidt Process to the columns of A
  U = gram_schmidt([Vec(row) for row in A.colsp]) #!!!!!!!!! # FIXME: Replace with the appropriate line################################
  n = len(U)

  # Constructing Q
  # Q should be the matrix whose columns are the elements
  # of the vector in set U
  Q = Matrix([[None for j in range(n)] for i in range(n)])
  for j in range(n):
    Q.set_col(j + 1, list(U[j]))  # FIXME: Replace with the appropriate line########################

  # Constructing R
  R = Matrix([[0 for j in range(n)] for i in range(n)])
  for j in range(n):
    for i in range(n):
      if i <= j:
        r = U[i] * Vec(A.get_col(j+1))
        R.set_entry(i + 1, j + 1, r) # # FIXME: Replace with the appropriate line#########################

  # Constructing the solution vector x
  b_star = Vec(Q.transpose() * b) #############og not vec
  x = [None for i in range(n)]
  for i in range(n-1,-1,-1): #og no n-1, just n, no -1,-1
  #!!!!!!!!!!!!!!!!!# FIXME: find the components of the solution vector and replace them into elements of x ##########################
    sum = b_star[i]
    for j in range( i + 1, n):
      sum = sum - R.rowsp[i][j] * x[j]
    x[i] = sum/R.rowsp[i][i]
  return Vec(x)


# ----------------------- PROBLEM 2 ----------------------- #
def _submatrix(A: Matrix, i: int, j: int):
  """
    constructs the sub-matrix of an mxn Matrix A that
    results from omitting the i-th row and j-th column;
    i and j satisfy that 0 <= i <= m, and 0 <= j <= n
    :param A: Matrix object
    :param i: int index of row to omit
    :param j: int index of column to omit
    :return: Matrix object representing the sub-matrix
    """
  m, n = A.dim()
  rows = []
   #FIXME: Implement this function##########################################
  if (0<= i and i<= m) and (0<=j and j<=n):
    rows = [row[:j-1]+row[j:] for row in A.rowsp[:i-1]+A.rowsp[i:]]
  return Matrix(rows)


# ----------------------- PROBLEM 3 ----------------------- #
def determinant(A: Matrix):
  """
    computes the determinant of square Matrix A;
    Raises ValueError if A is not a square matrix.
    :param A: Matrix object
    :return: float value of determinant
    """
  m, n = A.dim()
  if m != n:
    raise ValueError(
        f"Determinant is not defined for Matrix with dimension {m}x{n}.  Matrix must be square."
    )
  if n == 1:
    return A.get_entry(1,1)  # FIXME: Return the correct value#########################
  elif n == 2:
    return A.get_entry(1,1) * A.get_entry(2,2) - A.get_entry(1,2) * A.get_entry(2,1)  # FIXME: Return the correct value##########################
  else:
    d = 0
    # FIXME: Update d so that it holds the determinant
    #        of the matrix.  HINT: You should apply a
    #        recursive call to determinant()################################
    for j in range(n):
      d = d + ((-1)**j)*A.get_entry(1,j+1) * determinant(_submatrix(A,1,j+1)) #
    return d


# ----------------------- PROBLEM 4 ----------------------- #
def eigen_wrapper(A: Matrix):
  """
    uses numpy.linalg.eig() to create a dictionary with
    eigenvalues of Matrix A as keys, and their corresponding
    list of eigenvectors as values.
    :param A: Matrix object
    :return: Python dictionary
    """# !!!!!!FIXME: Implement this function#################################
  eig_val,eig_vec = np.linalg.eig(A.rowsp)
  e_dict = {}
  for i in range(len(eig_val)):
    eig_value = eig_val[i]
    eig_vect = eig_vec[:,i]
    e_dict[eig_value] = Vec(eig_vect)
  return e_dict


# ----------------------- PROBLEM 5 ----------------------- #
def svd(A: Matrix):
  """
    computes the singular value decomposition of Matrix A;
    returns Matrix objects U, Sigma, and V such that
        1. V is the Matrix whose columns are eigenvectors of 
        A.transpose() * A
        2. Sigma is a diagonal Matrix of singular values of 
        A.transpose() * A appearing in descending order along 
        the main diagonal
        3. U is the Matrix whose j-th column uj satisfies 
        A * vj = sigma_j * uj where sigma_j is the j-th singular value in 
        decreasing order and vj is the j-th column vector of V
        4. A = U * Sigma * V.transpose()
    :param A: Matrix object
    :return: tuple with Matrix objects; (U, Sigma, V)
    """
  m, n = A.dim()
  aTa = A.transpose() * A
  eigen = eigen_wrapper(aTa)
  eigenvalues = np.sort_complex(list(eigen.keys())).tolist()[::-1]

  # Constructing V
  # V should be the mxm matrix whose columns
  # are the eigenvectors of matrix A.transpose() * A
  V = Matrix([[None for j in range(n)] for i in range(n)])
  for j in range(1, n + 1):
    # !!!!!!!!FIXME: Replace this with the lines that will correctly build the entries of V####################################################
    eig_val = eigenvalues[j-1]
    eig_vec = eigen[eig_val]
    for i in range(1, n+1):
      V.set_entry(i,j,eig_vec[i-1])
  # Construct
    

  # Constructing Sigma
  # Sigma should be the mxn matrix of singular values.
  singular_values = np.sqrt(np.sort(np.abs(eigenvalues))[::1])  # FIXME: Replace this so that singular_values
  #        holds a list of singular values of A
  #        in decreasing order
  Sigma = Matrix([[0 for j in range(n)] for i in range(m)])
  for i in range(1, m + 1):
      # FIXME: Replace this with the lines that will correctly
    #        build the entries of Sigma############################################
    Sigma.set_entry(i,i,singular_values[i-1]) #????????????????????????????

  # Constructing U
  # U should be the matrix whose j-th column is given by
  # A * vj / sj where vj is the j-th eigenvector of A.transpose() * A
  # and sj is the corresponding j-th singular value
  U = Matrix([[None for j in range(m)] for i in range(m)])
  for j in range(1, m + 1):
      # FIXME: Replace this with the lines that will
    #        correctly build the entries of U##################################
    eig_val = eigenvalues[j-1]
    eig_vec = eigen[eig_val]
    sing_val = singular_values[j-1] #??????????????????????????
    ui = eig_vec/float(sing_val)
    uj = A*Vec(ui)
    U.set_col(j, list(uj))
  return (U, Sigma, V)
