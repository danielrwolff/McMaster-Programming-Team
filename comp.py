#Basic Graph Setup
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


#custom sorting comparison
def compare(x, y):
    return x - y
print sorted([5, 2, 4, 1, 3], cmp=compare)


#BFS
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


#DFS
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


#cc BFS
cc = {}
countc = 1
covered = set([])
for no in range(1,n+1):
    if no not in covered:
        q = Queue()
        covered.add(no)
        q.put(no)

        while not q.empty():
            current = q.get()
            cc[current] = countc
            nxt = []
            if current in gra2:
                nxt = gra2[current]
            for node in nxt:
                if node not in covered:
                    covered.add(node)
                    q.put(node)           
        countc += 1


#Bouncing Iteration through continguous subarrays
        n,q = map(int,raw_input().split())
a = list(map(int,raw_input().split()))

quer = {}
pairs = {}
for Q in range(q):
    u,v = map(int,raw_input().split())
    quer[(u,v)] = 0
    pairs[u] = v
    pairs[v] = u
    

count = 0
forward = True
freq = {}
s,e = -1,-1
while count < n*(n+1)/2:
    if forward:
        if e < n-1:
            #Add new
            e += 1
            if a[e] in freq:
                freq[a[e]] += 1
            else:
                freq[a[e]] = 1
            
            #Remove old
            s += 1
            if s-1>=0:
                freq[a[s-1]] -= 1
        else:
            #Change Direction
            forward = False
            s -= 1
            freq[a[s]] += 1
        
    else:
        if s > 0:
            #Add new
            s -= 1
            freq[a[s]] += 1
            
            #Remove old
            e -= 1
        else:
            #Change Direction
            forward = True
            e += 1
            freq[a[e]] += 1

    count += 1

#sum of all congruent subarray lengths
def f(n):
    return pow(n,3)/6.0 + pow(n,2)/2.0 + n/3.0

#sum of all subarray lengths
def g(n):
    return n*pow(2,n-1)


#iterate through all pairs in sequence
for i in range(1,n):
    for j in range(i+1,n+1):
        pass


#insertion sort
def insertion_sort(a):
    for j in range(1,len(a)):
        key = a[j]
        i = j-1
        while i>=0 and key < a[i]:
            a[i+1] = a[i]
            i -= 1
        a[i+1] = key
    return a


#selection sort
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


#merge sort
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

#quicksort
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

#radix sort
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



#heap
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


#----- Segment Tree -----
        
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


