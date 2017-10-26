

class Interval :
	def __init__(self, low, high) :
		self.low = low
		self.high = high

class Node :
	def __init__(self, interval, maxx, left, right) :
		self.interval = interval
		self.max = maxx
		self.left = left
		self.right = right

def newNode(interval) :
	return Node(interval, interval.high, None, None)

def insert(root, interval) :
	if not root : return newNode(interval)

	l = root.interval.low
	if interval.low < l :
		root.left = insert(root.left, interval)
	else :
		root.right = insert(root.right, interval)

	root.max = max(root.max, interval.high)

	return root

def doOverlap(inter1, inter2) :
	if inter1.low <= inter2.high and inter2.low <= inter1.high : return True
	return False

def overlapSearch(root, interval) :
	if not root : return None

	if doOverlap(root.interval, interval) : return root.interval

	if root.left is not None and root.left.max >= interval.low :
		return overlapSearch(root.left, interval)

	return overlapSearch(root.right, interval)

def inOrder(root) :
	if not root : return None
	inOrder(root.left)
	print "[",root.interval.low,",",root.interval.high,"]",root.max
	inOrder(root.right)

# USAGE

intervals = [	[15, 20],
				[10, 30],
				[17, 19],
				[5, 20],
				[12, 15],
				[30, 40]
			]

root = None
for i in range(len(intervals)) :
	root = insert(root, Interval(intervals[i][0], intervals[i][1]))

inOrder(root)

x = Interval(6,7)
res = overlapSearch(root, x)
if not res : print "No overlap!"
else : print "Overlap with [",res.low,",",res.high,"]"

