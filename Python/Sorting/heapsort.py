def left(i):
    return 2*i+1

def right(i):
    return 2*i+2

def max_heapify(A, heap_size, i):
        l = left(i)
        r = right(i)

        if l < heap_size and A[l] > A[i]:
            largest = l
        else:
            largest = i
        if r < heap_size and A[r] > A[largest]:
            largest = r

        if not largest == i:
            A[i], A[largest] = A[largest],A[i]

            max_heapify(A, heap_size, largest)

def build_max_heap(A):
    heap_size = len(A)
    for i in xrange(int(len(A)/2)-1,-1,-1):
        max_heapify(A,heap_size,i)

def heap_sort(A):
    build_max_heap(A)
    heap_size = len(A)
    for i in xrange(len(A)-1,0,-1):
        A[0],A[i] = A[i],A[0]
        heap_size -= 1
        max_heapify(A, heap_size, 0)
        


import random
a = [int(random.random()*10) for k in range(1+int(random.random()*15))]

print a
heap_sort(a)

print a


