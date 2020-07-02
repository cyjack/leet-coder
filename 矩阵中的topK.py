def kthSmallest(matrix, k):
    """
    :type matrix: List[List[int]]
    :type k: int
    :rtype: int
    """
    list = []
    for i in matrix:
        for j in i:
            list.append(j)

    for i in range(len(list)):
        for j in range(len(list)):
            if list[j] >list[i]:
                temp = list[j]
                list[j] = list[i]
                list[i] =temp
                print(sum(matrix, []))
    print(list)
    return list[k-1]

    #return sorted(sum(matrix, []))[k-1] #暴力解法最快的方式







matrix = [[1,2,3],
          [7,8,9],
          [9,10,11]]
print(kthSmallest(matrix,8))