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
