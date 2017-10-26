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



