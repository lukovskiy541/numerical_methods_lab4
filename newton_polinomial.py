import numpy as np
import matplotlib.pyplot as plt

def newton_polynomial(x, x_nodes, coef):
    n = len(coef)
    result = coef[0, 0]
    product = 1.0
    for i in range(1, n):
        product *= (x - x_nodes[i - 1])
        result += coef[0, i] * product
    return result

def divided_differences(x_nodes, y_nodes):
    n = len(x_nodes)
    coef = np.zeros([n, n])
    coef[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x_nodes[i + j] - x_nodes[i])
    return coef

def newton_polynomial_expression(x_nodes, coef):
    n = len(coef)
    terms = [f"{coef[0, 0]:.6f}"] 
    for i in range(1, n):
        if x_nodes[i-1] < 0 :
            term = f"(x + { -x_nodes[i-1]})"
        else :
            term = f"(x - {x_nodes[i-1]})"
        
        
        for j in range(i-1):
            if x_nodes[j] < 0 :
                term = f"(x + {-x_nodes[j]})*{term}"
            else :
                term = f"(x - {x_nodes[j]})*{term}"
            
        terms.append(f"{coef[0, i]:.6f}*{term}")  
    
    return " + ".join(terms)

a, b = -1, 1
f = lambda x: np.exp(x) + np.arctan(x)
print("Введіть n:")
n=int(input())
x_nodes = np.linspace(a, b, n)
y_nodes = f(x_nodes)
print(f"x_k = {[float(x) for x in x_nodes]}")
print(f"f(x_k) = {[float(f(x)) for x in x_nodes]}")

h = (b - a)/2
print(f"h = {h}")

coef = divided_differences(x_nodes, y_nodes)

print("\nРозділені різниці:")
for j in range(1, n):
    print(f"p.p. {j} п.: " + " ".join([f"{coef[i, j]:.6f}" for i in range(n-j)]))

polynomial_expression = newton_polynomial_expression(x_nodes, coef)
print(f"1) Вигляд поліному Ньютона: {polynomial_expression}")

x_values = np.linspace(a - 3, b + 3, 100)
f_values = f(x_values)
P_values = [newton_polynomial(x, x_nodes, coef) for x in x_values]
difference_values = f_values - np.array(P_values)

y_min = np.min([f(a), f(b)])
y_max = np.max([f(a), f(b)])

y_star = y_min + (1 / 3) * (y_max - y_min)

print(f"y* = {y_star:.6f}")

coef = divided_differences(y_nodes, x_nodes)

print("\nРозділені різниці:")
for j in range(1, n):
    print(f"p.p. {j} п.: " + " ".join([f"{coef[i, j]:.6f}" for i in range(n-j)]))
    
polynomial_expression = newton_polynomial_expression(y_nodes, coef)
print(f"Вигляд поліному Ньютона: {polynomial_expression}")

result = newton_polynomial(y_star, y_nodes, coef)

print(f"3) Результат обчислення: {result}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_values, f_values, label='f(x) = e^x + arctan(x)', color='blue')
plt.plot(x_values, P_values, label='P_n(x)', linestyle='--', color='red')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='Nodes (x_k, f(x_k))')
plt.title('f(x) та P_n(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_values, difference_values, label='f(x) - P_n(x)', color='green')
plt.title('f(x) - P_n(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()


