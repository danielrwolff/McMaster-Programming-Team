# Python Cheat Sheet
#### Written by McMaster's Competitive Programming Team
## ./Data_Structures
``` python
class Heap():
    def __init__(self,A):
        self.A = A
        self.make()

    def left(self,i): return 2*i+1
    
    def right(self,i): return 2*i+2
    
    def parent(self,i): return (i-1)/2
    
    def make(self):
        self.heap_size = len(self.A)
        for i in xrange(int(self.heap_size/2)-1,-1,-1):
            self.max_heapify(i)

    def max_heapify(self,i):
            l = self.left(i)
            r = self.right(i)

            if l < self.heap_size and self.A[l] > self.A[i]:
                largest = l
            else:
                largest = i
            if r < self.heap_size and self.A[r] > self.A[largest]:
                largest = r

            if not largest == i:
                self.A[i], self.A[largest] = self.A[largest],self.A[i]

                self.max_heapify(largest)

    def sort(self):
        for i in xrange(len(self.A)-1,0,-1):
            self.A[0],self.A[i] = self.A[i],self.A[0]
            self.heap_size -= 1
            self.max_heapify(0)

    def maximum(self):
        return self.A[0]

    def extract_max(self):
        assert self.heap_size > 0
        mx = self.A[0]
        self.A[0] = self.A[self.heap_size-1]
        self.heap_size -= 1
        self.max_heapify(0)
        return mx

    def increase_key(self,i,key):
        assert key >= self.A[i]
        self.A[i] = key
        while i>0 and self.A[i] > self.A[self.parent(i)]:
            self.A[i], self.A[self.parent(i)] = self.A[self.parent(i)], self.A[i]
            i = self.parent(i)

    def insert_key(self,key):
        self.heap_size += 1
        self.A.append(key)
        self.increase_key(self.heap_size-1,key)
        



import random
a = [int(random.random()*10) for k in range(1+int(random.random()*15))]



```
``` python
# Python implementation of Binary Indexed Tree - GEEKSFORGEEKS
 
# This code is contributed by Raju Varshney

# Returns sum of arr[0..index]. This function assumes
# that the array is preprocessed and partial sums of
# array elements are stored in BITree[].
def getsum(BITTree,i):
    s = 0  #initialize result
 
    # index in BITree[] is 1 more than the index in arr[]
    i = i+1
 
    # Traverse ancestors of BITree[index]
    while i > 0:
 
        # Add current element of BITree to sum
        s += BITTree[i]
 
        # Move index to parent node in getSum View
        i -= i & (-i)
    return s
 
# Updates a node in Binary Index Tree (BITree) at given index
# in BITree.  The given value 'val' is added to BITree[i] and
# all of its ancestors in tree.
def updatebit(BITTree , n , i ,v):
 
    # index in BITree[] is 1 more than the index in arr[]
    i += 1
 
    # Traverse all ancestors and add 'val'
    while i <= n:
 
        # Add 'val' to current node of BI Tree
        BITTree[i] += v
 
        # Update index to that of parent in update View
        i += i & (-i)
 
 
# Constructs and returns a Binary Indexed Tree for given
# array of size n.
def construct(arr, n):
 
    # Create and initialize BITree[] as 0
    BITTree = [0]*(n+1)
 
    # Store the actual values in BITree[] using update()
    for i in range(n):
        updatebit(BITTree, n, i, arr[i])
 
    # Uncomment below lines to see contents of BITree[]
    #for i in range(1,n+1):
    #      print BITTree[i],
    return BITTree
 
 
# Driver code to test above methods
freq = [2, 1, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9]
BITTree = construct(freq,len(freq))
print("Sum of elements in arr[0..5] is " + str(getsum(BITTree,5)))
freq[3] += 6
updatebit(BITTree, len(freq), 3, 6)
print("Sum of elements in arr[0..5] is " + str(getsum(BITTree,5)))
 ```
``` python


class Interval :
	def __init__(self, low, high) :
		self.low = low
		self.high = high

class Node :
	def __init__(self, interval, maxx, left, right) :
		self.interval = interval
		self.max = maxx
		self.left = left
		self.right = right

def newNode(interval) :
	return Node(interval, interval.high, None, None)

def insert(root, interval) :
	if not root : return newNode(interval)

	l = root.interval.low
	if interval.low < l :
		root.left = insert(root.left, interval)
	else :
		root.right = insert(root.right, interval)

	root.max = max(root.max, interval.high)

	return root

def doOverlap(inter1, inter2) :
	if inter1.low <= inter2.high and inter2.low <= inter1.high : return True
	return False

def overlapSearch(root, interval) :
	if not root : return None

	if doOverlap(root.interval, interval) : return root.interval

	if root.left is not None and root.left.max >= interval.low :
		return overlapSearch(root.left, interval)

	return overlapSearch(root.right, interval)

def inOrder(root) :
	if not root : return None
	inOrder(root.left)
	print "[",root.interval.low,",",root.interval.high,"]",root.max
	inOrder(root.right)

# USAGE

intervals = [	[15, 20],
				[10, 30],
				[17, 19],
				[5, 20],
				[12, 15],
				[30, 40]
			]

root = None
for i in range(len(intervals)) :
	root = insert(root, Interval(intervals[i][0], intervals[i][1]))

inOrder(root)

x = Interval(6,7)
res = overlapSearch(root, x)
if not res : print "No overlap!"
else : print "Overlap with [",res.low,",",res.high,"]"

```
## ./strings
## ./Useful_Tidbits
## ./Graph_Theory
``` python
def disjoint_set():
    def find(x):
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(x,y):
        x_root = find(x)
        y_root = find(y)
        parents[x_root] = y_root

    nodes = 8
    parents=range(nodes)

    while True:
        print parents
        x,y = map(int,raw_input().split())
        union(x,y)
```
## ./Dynamic_Programming
## ./Sorting
``` python
def do_quicksort(A):
    quicksort(A,0,len(A)-1)
    return A

def quicksort(A,p,r):
    if p<r:
        q = partition(A,p,r)
        quicksort(A,p,q-1)
        quicksort(A,q+1,r)

def partition(A,p,r):
    x = A[r]
    i = p-1
    for j in range(p,r):
        if A[j] <= x:
            i += 1
            A[i],A[j] = A[j],A[i]
    A[i+1],A[r] = A[r],A[i+1]
    return i+1


#init random numbers for testing
import random
li = [int(random.random()*10) for k in range(1+int(random.random()*15))]
print "Unsorted"
print li

print "Sorted"
print do_quicksort(li)
```
``` python
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
```
``` python
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


```
``` python
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
```
``` python
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
```
``` python
def selection_sort(a):
    for i in range(len(a)-1):
        small = a[i]
        ind = i
        for j in range(i+1,len(a)):
            if a[j]<small:
                small = a[j]
                ind = j
        a[i],a[ind] = a[ind],a[i]
    return a
        

#init random numbers for testing
import random
li = [int(random.random()*10) for k in range(1+int(random.random()*15))]
print "Unsorted"
print li

print "Sorted"
print selection_sort(li)
```
``` python
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
```
