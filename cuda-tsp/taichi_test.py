import taichi as ti

ti.init(arch=ti.cpu)

lst = ti.field(dtype=ti.i32, shape=20)

a = ti.field(ti.i32,shape=(5,5))
b = ti.field(ti.i32,shape=(5,5))
c = [1,2,3,4,5]

@ti.kernel
def test(a:ti.template()):
    for i in range(5):
        check(i)

@ti.func
def check(x: int):
    for j in range(5):
        a[x,j] = c[a[x,j]]

for i in range(5):
    for j in range(5):
        a[i,j] = i+j

print(a)
test(a)
print(a)