class Solution(object):
    def recursive_fast_order(self, nums):
        if (len(nums) >= 2):
            mid = nums[len(nums) // 2]
            right, left = [], []
            del nums[len(nums) // 2]
            for items in nums:
                if items >= mid:
                    left.append(items)
                else:
                    right.append(items)
            return self.recursive_fast_order(left) + [mid] + self.recursive_fast_order(right)
        else:
            return nums


    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        nums = self.recursive_fast_order(nums)
        print(nums[k-1])
        return nums[k-1]
nums = [1,2,6,4,3,6,8,3,56]
a= Solution()
a.findKthLargest(nums,5)