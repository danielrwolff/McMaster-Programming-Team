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
