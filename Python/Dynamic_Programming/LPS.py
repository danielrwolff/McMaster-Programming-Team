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
