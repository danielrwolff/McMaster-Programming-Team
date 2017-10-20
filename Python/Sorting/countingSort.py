def counting_sort(A,k):
    C = [0]*(k+1)
    for ele in A:
        C[ele] += 1
    for i in xrange(1,k+1):
        C[i] = C[i] + C[i-1]
    B = [0]*len(A)
    for i in xrange(len(A)-1,-1,-1):
        B[C[A[i]]-1] = A[i]
        C[A[i]] -= 1
    return B

    
#init random numbers for testing
import random
li = [int(random.random()*10) for k in range(1+int(random.random()*15))]
print "Unsorted"
print li

print "Sorted"
print counting_sort(li,max(li))
