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
