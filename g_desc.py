

def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 -7)**2

def grad(x, y):
    gradx = 4*x**3 + 4*x*y - 42*x + 2*y**2 - 14
    grady = 2*x**2 + 4*x*y + 4*y**3 - 26*y - 22
    return gradx, grady


x = 0
y = 0
lr = 0.01
for a in range(100):
    loss = f(x, y)
    dx, dy = grad(x,y)
    print(f"loss: {loss}")
    print(f"dx: {dx}, dy: {dy}")

    x -= lr*dx
    y -= lr*dy
    print(f"x:{x}, y:{y}")

    if loss - f(x,y) < 0.001:
        print("final: ", x, y, a)
        break

