def addBinary( a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    i = len(a ) -1
    j = len(b ) -1
    cout = 0
    z = ''
    while i>= 0 or j >= 0 or cout:
        x = int(a[i]) if i >= 0 else 0
        y = int(b[j]) if j >= 0 else 0
        cout, res = divmod(x + y + cout, 2)
        z = str(res) + z
        i -= 1
        j -= 1
        print(i,j)
    return z


print(addBinary('100','1101'))