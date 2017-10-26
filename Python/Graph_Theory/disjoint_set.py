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
