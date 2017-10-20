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
