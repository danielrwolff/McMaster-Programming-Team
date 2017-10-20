def bucket_sort(A):
    n = len(A)
    B = [[] for i in xrange(n)]
    R = []
    for ele in A:
        B[int(n*A[i])].append(ele)
    for b in B:
        #insertion sort
        for j in range(1,len(b)):
            key = b[j]
            i = j-1
            while i>=0 and key < b[i]:
                b[i+1] = b[i]
                i -= 1
            b[i+1] = key
        R += b
    return R
        

#init random numbers for testing
import random
li = [random.random() for k in range(1+int(random.random()*15))]
print "Unsorted"
print li

print "Sorted"
print bucket_sort(li)
