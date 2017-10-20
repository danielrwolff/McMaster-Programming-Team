def counting_sort(A,k,dig):
    div = 10**dig
    C = [0]*(k+1)
    for ele in A:
        C[(ele/div)%10] += 1
    for i in xrange(1,k+1):
        C[i] = C[i] + C[i-1]
    B = [0]*len(A)
    for i in xrange(len(A)-1,-1,-1):
        B[C[(A[i]/div)%10]-1] = A[i]
        C[(A[i]/div)%10] -= 1
    return B

def radix_sort(A,d):
    for i in range(d):
        A = counting_sort(A,9,i)
    return A
#init random numbers for testing
import random
li = [int(random.random()*30) for k in range(1+int(random.random()*15))]
print "Unsorted"
print li

print "Sorted"
print radix_sort(li,2)
