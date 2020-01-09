class test:
    def __init__(self):
        self.a = 1
        exec("self.b = '2'")


b = test()

print(b.a)

print(b.b)

print(type(b.b))

