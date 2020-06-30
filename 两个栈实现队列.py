class CQueue:

    def __init__(self):
        self.A = []
        self.B = []


    def appendTail(self, value: int) -> None:
        self.A.append(value)


    def deleteHead(self) -> int:
        # 如果B 栈中含有数据则将栈顶数据出栈
        if self.B :
            return self.B.pop()
        # 如果A，B栈都为空，则直接返回-1
        if not self.A :
            return -1
        # 如果B栈空，A栈有数据，则将A栈的数据从出栈放入到B栈中
        while( self.A ):
            self.B.append(self.A.pop())
        return self.B.pop()

c = CQueue()
c.appendTail('1')
c.appendTail('2')
c.appendTail('3')
print(c.A)
c.deleteHead()
print(c.A)
print(c.B)