
def findLength(A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        dp =[ [0 for i in range(len(A)+1)] for i in range(len(B)+1) ]
        ans =0
        for i in range(1,len(B)):
            for j in range(1,len(A)):
                if(A[j-1] == B[i-1]):
                    dp[i][j] = dp[i-1][j-1]+1
                    ans = max(ans, dp[i][j])
        # max = 0
        # for i in range(0, len(B)+1):
        #     for j in range(0, len(A)+1):
        #         if dp[i][j] >max:
        #             max = dp[i][j]
        return ans

A = [0,2,3,4,5]
B = [2,3,4,0,2,3,4]
print(findLength(A,B))