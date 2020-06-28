
def minSubArraylen(s,nums):
    minlen = len(nums)
    for i in range(len(nums)):
        temp = 0
        lentemp = 0
        for j in range(i,len(nums)):
            temp = temp +nums[j]
            lentemp+=1
            if temp < s and j == len(nums)-1:
                print(minlen)
                exit(0)
            elif temp >= s and lentemp <= minlen :
                    minlen = lentemp
                    break
    if minlen == len(nums):
        print(0)
    else:
        print(minlen)
minSubArraylen(10,[5,4,1,9,7,6,9,9])


