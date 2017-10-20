import sys
def merge_sort(a):
    mid = len(a)/2
    if len(a) == 1 or len(a) == 0:
        return a[:]
    left = merge_sort(a[:mid])
    right = merge_sort(a[mid:])
    return merge(left,right)

def merge(A,B):
    A.append(sys.maxint)
    i = 0
    B.append(sys.maxint)
    j = 0
    res = []
    for k in xrange(len(A)+len(B)-2):
        if A[i]<B[j]:
            res.append(A[i])
            i += 1
        else:
            res.append(B[j])
            j += 1
    return res
    




        

#init random numbers for testing
import random
li = [int(random.random()*10) for k in range(1+int(random.random()*15))]
print "Unsorted"
print li

print "Sorted"
print merge_sort(li)
