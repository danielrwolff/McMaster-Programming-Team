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
