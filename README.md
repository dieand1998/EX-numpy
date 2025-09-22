# EX-numpy

## 100 Ejercicios en numpy

Después de clase, revisa este cuaderno con [100 ejercicios cortos de numpy](https://colab.research.google.com/github/jyothi870/tejaswini5g5/blob/main/assignments/100_Numpy_exercises.ipynb). 

Por ahora, resuelve el siguiente ejercicio. 

## Ejercicio en clase

- Escriba una función que reciba una matriz $X$, de tamaño $n\times p$ y un vector columna $y$ de tamaño $n\times 1$ para computar los coeficientes estimados $\hat\beta$ del modelo de regresión lineal $y = X\beta+u$. 
- La matrix $X$ no contiene la columna de unos, debe ser agregada en la función para estimar el intercepto de la regresión lineal y conformar correctamente la matriz de diseño.
- Utilice operaciones de álgebra matricial de NumPy. 


```python
def ols_estimation(X: np.ndarray, y:np.ndarray): 
    beta_hat = np.zeros((X.shape[1]+1, 1))
    return beta_hat
```


```python
# Carga de datos para una regresión
admission = np.load("admission.npy")
X = admission[:, :-1]
y = admission[:, -1][:, None]
X.shape, y.shape
```




    ((400, 8), (400, 1))




```python
betahat = ols_estimation(X, y)
betahat
```




    array([[-1.29364902e+00],
           [ 1.59313401e-04],
           [ 1.79901054e-03],
           [ 3.68217418e-03],
           [ 8.78497695e-03],
           [ 9.93694120e-05],
           [ 2.15369726e-02],
           [ 1.05298032e-01],
           [ 2.43772863e-02]])
