# < Fast Fibonacci >
from math import sqrt

def fib(n):
    sqr5 = sqrt(5)
    a = (1 + sqr5)/2
    b = (1 - sqr5)/2
    return int((a**n-b**n)/sqr5)

# ---- Example Usage
print fib(1) # 1
print fib(2) # 1
print fib(3) # 2
print fib(4) # 3
print fib(5) # 5
print fib(6) # 8
