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
# < Segment Tree >     
import math

st = []

#Mid
def mid(s,e):
    return s + (e-s)/2

#Constructor
def ST(a,n):
    global st
    #Height of Tree
    h = math.ceil(math.log(n)/float(math.log(2)))
    max_size = 2*int(math.pow(2,h))-1

    st = [0]*max_size

    buildST(a,0,n-1,0)

#get Max
def getMax(ss,se,qs,qe,si):
    global st
    if qs <= ss and qe >= se:
        return st[si]

    if se < qs or ss > qe:
        return 0

    m = mid(ss,se)
    return max(getMax(ss,m,qs,qe,2*si+1),getMax(m+1,se,qs,qe,2*si+2))

#user friendly max :)
def niceMax(start,end,n):
    return getMax(0,n-1,start,end,0)


#get first ele over num
def overNum(ss,se,num,si):
    global st
    if st[si]<num:
        return -1
    
    if ss == se:
        return ss

    m = mid(ss,se)
    if st[2*si+1]>=num:
        return overNum(ss,m,num,2*si+1)
    else:
        return overNum(m+1,se,num,2*si+2) 
    
#Construction of Segment Tree
def buildST(a,ss,se,si):
    global st
    #one ele
    if ss==se:
        st[si] = a[ss]
        return a[ss]
    
    #multiple eles
    m = mid(ss,se)
    st[si] = max(buildST(a,ss,m,si*2+1),buildST(a,m+1,se,si*2+2))
    return st[si]


# ---- Example Usage
a = [5, 6, 2, 1, 3, 7, 10]
ST(a, len(a))
print niceMax(1,5,7) # Max element from a[1] -> a[5]
```
``` python
# < 2D Partial Sums >
from Queue import Queue

#Constructor
def build2dSum(n, m, a): #rows, cols, array
    d = [[0 for j in xrange(m)] for i in xrange(n)]

    d[0][0] = a[0][0]
    for i in xrange(1, n): d[i][0] = a[i][0] + d[i-1][0]
    for j in xrange(1, m): d[0][j] = a[0][j] + d[0][j-1]

    if n >= 2 and m >= 2:
        q = Queue()
        cov = set()
        q.put((1,1))

        while not q.empty():
            r,c = q.get()
            if (r,c) not in cov:
                cov.add((r,c))
                d[r][c] = a[r][c] + d[r-1][c] + d[r][c-1] - d[r-1][c-1]
                
                if r+1<n: q.put((r+1,c))
                if c+1<m: q.put((r,c+1))
    return d

def sumFrom(d,r1,c1,r2,c2):
    if r1 > 0 and c1 > 0:
        return d[r2][c2] - d[r2][c1-1] - d[r1-1][c2] + d[r1-1][c1-1]
    elif r1 > 0:
        return d[r2][c2] - d[r1-1][c2]
    elif c1 > 0:
        return d[r2][c2] - d[r2][c1-1]
    else:
        return d[r2][c2]


# --- Example Usage
a = [[1,1,1,1],
     [1,1,1,1],
     [1,1,1,1]]
T = build2dSum(3,4,a)
print sumFrom(T,0,0,0,0) #1
print sumFrom(T,0,0,2,3) #12
print sumFrom(T,1,0,1,3) #4
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
## ./Useful_Tidbits
``` python
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
```
``` python
# <Binary Search Recursive>
def binSearch(a, s, e, key):
    if s > e:
        return -1
    mid = (s + e)/2
    if a[mid] == key:
        return mid
    elif key < a[mid]:
        return binSearch(a, s, mid-1, key)
    elif key > a[mid]:
        return binSearch(a, mid+1, e, key)

# --- Example Usage
a = [1,5,17,22,35,70]
print binSearch(a, 0, len(a)-1, -5)  # -1
print binSearch(a, 0, len(a)-1, 1)   #  0
print binSearch(a, 0, len(a)-1, 7)   # -1
print binSearch(a, 0, len(a)-1, 70)  #  5
print binSearch(a, 0, len(a)-1, 999) # -1


# <Binary Search Iterative>
def binarySearch(a, s, e, key):
    res = -1
    while s <= e:
        mid = (s + e)/2
        if a[mid] == key:
            res = mid
            break
        elif key < a[mid]:
            e = mid - 1
        elif key > a[mid]:
            s = mid + 1
    return res

# --- Example Usage
a = [1,5,17,22,35,70]
print binarySearch(a, 0, len(a)-1, -5)  # -1
print binarySearch(a, 0, len(a)-1, 1)   #  0
print binarySearch(a, 0, len(a)-1, 7)   # -1
print binarySearch(a, 0, len(a)-1, 70)  #  5
print binarySearch(a, 0, len(a)-1, 999) # -1
```
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
## ./Strings
## ./Dynamic_Programming
``` python
# < Count Subarrays summing to 0 >
def countSubArrays(a, n):
    seen = {}

    rs = 0
    cnt = 0
    for i in xrange(n):
        rs += a[i]
        if rs == 0: cnt += 1

        if rs in seen:
            cnt += seen[rs]
            seen[rs] += 1
        else:
            seen[rs] = 1

    return cnt

# ---- Example usage
a = [6, 3, -1, -3, 4, -2, 2, 4, 6, -12, -7]
print countSubArrays(a, len(a))



# < Print Subbarrays summing to 0 >
def printSubArrays(a, n):
    seen = {}

    rs = 0
    out = []
    for i in xrange(n):
        rs += a[i]
        if rs == 0: out.append((0,i))

        if rs in seen:
            for old in seen[rs]:
                out.append((old+1,i))
            seen[rs].append(i)
        else:
            seen[rs] = [i]

    for pair in out:
        print pair

# ---- Example usage
a = [6, 3, -1, -3, 4, -2, 2, 4, 6, -12, -7]
printSubArrays(a, len(a))
```
``` python
# < Longest palindromic subsequence >
def LPS(a,n):
    fast = [[0 for x in xrange(n)] for y in xrange(n) ]
    for i in xrange(n): fast[i][i] = 1
    
    for size in xrange(2,n+1):
        for s in xrange(n-size+1):
            e = s+size-1
            if a[s] == a[e] and size == 2:
                fast[s][e] = 2
            elif a[s] == a[e]:
                fast[s][e] = fast[s+1][e-1] + 2
            else:
                fast[s][e] = max(fast[s][e-1],fast[s+1][e])
    print fast[0][n-1]

# --- Example usage
string = raw_input()
print LPS(string,len(string))
```
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
