import taichi as ti

ti.init(arch=ti.cpu)

lst = ti.field(dtype=ti.i32, shape=20)

@ti.kernel
def test(a:ti.template()):
    for i in range(10):
        # check(i)
        print(a)

@ti.func
def check(x):
    a = int(ti.random()*10)
    print(a)

for i in range(3):
    test(i)