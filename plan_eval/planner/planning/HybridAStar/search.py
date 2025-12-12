from bisect import bisect_left

def takeClosest(myList, myNumber):
    """
     Assumes myList is sorted. Returns closest value to myNumber.
     If two numbers are equally close, return the smallest number.
     If number is outside of min or max return False
    """
    if (myNumber > myList[-1] or myNumber < myList[0]):
        return False
    pos = bisect_left(myList, myNumber)
    return pos

if __name__ == "__main__":
    a = [1,4,5,7,31,42,112,175,198,465]
    b = 6
    answer = takeClosest(a, b)
    print(answer)
