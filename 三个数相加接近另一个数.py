def threeSumClosest( nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] < nums[j]:
                temp = nums[i]
                nums[i] = nums[j]
                nums[j] = temp
    print(nums)
    a = []
    for i in range(len(nums) - 2):
        a.append(nums[i] + nums[i + 1] + nums[i + 2])
    print(a)
    b =0
    for i in range(len(a)):

        if target >= a[i] and target <= a[i + 1]:
          print("我执行了")

          if ((target - a[i]) > (a[i+1] - target)) :
              print("我也执行了")
              b = a[i+1]
          else :
              b = a[i]
          print(b)
          break

print(threeSumClosest([9,2,3,4,1,3],10))