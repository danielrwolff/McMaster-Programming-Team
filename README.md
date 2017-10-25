# McMaster-Programming-Team

A collaborative repo for McMaster University's competitive programming teams.

# Python Cheat sheet

## Data Structures

## Dynamic Programming

## Graph Theory

Basic Graph Setup
``` Python
gra = {}
for edge in range(n-1):
    u,v = map(int,raw_input().split())
    if u not in gra:
        gra[u] = [v]
    else:
        gra[u].append(v)
        
    if v not in gra:
        gra[v] = [u]
    else:
        gra[v].append(u)

```
Union-Find Disjoint Set
```Python
def disjoint_set():
    def find(x):
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(x,y):
        x_root = find(x)
        y_root = find(y)
        parents[x_root] = y_root

nodes = int(raw_input())
parents=range(nodes)

while True:
    x,y = map(int,raw_input().split())
    union(x,y)
```

BFS
``` Python
from Queue import Queue
q = Queue()
covered = set([s])
q.put(s)

while not q.empty():
    cur = q.get()
    nxt = []
    if cur in gra:
        nxt = gra[cur]
    for node in nxt:
        if node not in covered:
            covered.add(node)
            q.put(node)

```

DFS
``` Python
q = []
covered = set([s])
q.append(s)
connected = False

while q:
    cur = q.pop()
    if cur==t:
        connected = True
        break
    nxt = []
    if cur in gra:
        nxt = gra[cur]
    for node in nxt:
        if node not in covered:
            covered.add(node)
            q.append(node)

```

## Sorting

Custom Sorting Comparision
``` Python
def compare(x, y):
    return x - y
print sorted([5, 2, 4, 1, 3], cmp=compare)
```

Selection Sort
```Python
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

Merge Sort
```Python
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
Quick Sort
```Python

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

Radix Sort
```Python

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

## Strings
