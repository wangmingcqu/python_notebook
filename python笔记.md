## 一、基本语法

```python
# 1、判断语句none也是可以的，只会输出第三个。
if None:
    print("nihao")  # 不会输出

if False:
    print("nihao")	# 不会输出

if True:
    print("nihao")	# 可以输出
```



### 1、基本数据类型

```python
5、Python中一切皆对象。不管是我们自己定义的类、6种基本数据类型还是内置的类(第三方类)等都称作为对象
 一个对象，就会拥有自己的属性和方法。我们可以通过一定的方式来调用一个对象的属性和方法。这一点自定义类与其他Python类型是一样的
属性就是类中定义的值，方法就是类中定义的值
    
#python中的6大基本数据类型
Numbers（数字）
String（字符串）
List（列表）


Tuple（元组）  也可以称为序列sequence
Dictionary（字典）
Set（集合）

在每一种数据类型中都定义了相应的方法


数字、字符串和元祖是不可变数据类型，后几种为可变数据类型
内置函数 type(), 用以查询变量的类型
print(type(nihao))#nihao是自己定义的数据类型

集合和字典是无序的，列表与元祖是有序的，列表和元祖唯一的区别就是元祖一旦创建就不可以被修改


print(type(5)) # <class 'int'>
print(type(5.)) # <class 'float'>



b=(1,2,3)
print(*b)
<<<1 2 3 #加上*就是将这个元祖给分解了


# python中的位运算符
&符号在Python中既可以执行通常的按位与运算，也可以执行set集合里面的交集运算
|：并集；也可以表示数字运算中的按位或运算
-：差集
^：对称差集
```

#### 1.1数字

```python
int(整型), float（浮点型）, bool, complex（复数类型） 四种基本类型，用于存储数值
```

#### 1.2字符串

字符串需要用单引号 ’ ’ 或双引号 " " 括起来 三引号–注释
字符串也是一种特殊的元组。不能改变字符串中的某个元素的值

```python
# 1、
#!/usr/bin/env python3
str1 = 'Hello'
print(str1[0])

#字符串格式化操作
a="我叫%s" % "王明"
print(a)
>>>我叫王明

# 输出整数也是可以的,下面是输出多个类型的
print '%s %d %f'%('jack',29,9.2)
>>>jack 29 9.200000

```

1、**字符串startswith()方法与endswith()方法** 

`startswith()`用于检查一个字符串是否以某一个子串开头,是的话返回True，否则返回False

```python
str.startswith(prefix, [,start [,end ])
```

startswith() 方法接受三个参数：

- prefix 是一个需要查找的字符串或者字符串元组。
- start 是字符串 str 中查找操作开始的位置。这是一个可选的参数。
- end 是字符串 str 中查找操作结束的位置。这也是一个可选的参数。

```python
s = 'Wangming love shaolin'
result = s.startswith('Wang')
print(result)  # True

result = s.startswith('wang')
print(result)  # False 表示区分大小写

result = s.startswith(('Wang','wang')) # 输入元祖
print(result) # True

result = s.startswith('love',9) # 索引是从0开始的
print(result) # True
```



`endswith()`方法检测一个字符串是否以某个子串结束。如果是，返回 True；否则，返回 False。

```python
str.endswith(suffix, [,start [,end ])
```

endswith() 方法接受三个参数：

- suffix 是一个需要查找的字符串或者字符串元组。
- start 是字符串 str 中查找操作开始的位置。这是一个可选的参数。
- end 是字符串 str 中查找操作结束的位置。这也是一个可选的参数。

```python
marks = ('.', '?', '!')
sentence = 'Hello, how are you?'
result = sentence.endswith(marks)
print(result) #  True
```

2、字符串填充函数str.zfill()

```python
# 字符串填充函数,里面的数字表示字符串总的宽度
b = "0".zfill(3)
print(type(b))    # <class 'str'>
print(b)          # 000

a = "a".zfill(5)
print(a)          # 0000a
```



```python
"terminal.background":"#090300",
"terminal.foreground":"#A5A2A2",
"terminalCursor.background":"#A5A2A2",
"terminalCursor.foreground":"#A5A2A2",
"terminal.ansiBlack":"#090300",
"terminal.ansiBlue":"#01A0E4",
"terminal.ansiBrightBlack":"#5C5855",
"terminal.ansiBrightBlue":"#01A0E4",
"terminal.ansiBrightCyan":"#B5E4F4",
"terminal.ansiBrightGreen":"#01A252",
"terminal.ansiBrightMagenta":"#A16A94",
"terminal.ansiBrightRed":"#DB2D20",
"terminal.ansiBrightWhite":"#F7F7F7",
"terminal.ansiBrightYellow":"#FDED02",
"terminal.ansiCyan":"#B5E4F4",
"terminal.ansiGreen":"#01A252",
"terminal.ansiMagenta":"#A16A94",
"terminal.ansiRed":"#DB2D20",
"terminal.ansiWhite":"#A5A2A2",
"terminal.ansiYellow":"#FDED02"
```







#### 1.3列表

- list 的数据项可以不同类型
- list 的各个元素可以改变
- list 是使用 [ ] 方括号包含各个数据项
- 使用[ ]或者list来创建

```python
#!/usr/bin/env python3
list1 = [True, 1, 'Hello']
print(list1)


# 注意列表元素相加并不是对应的位置相加，而是往后面继续补充，可以不相同
a=[1,2,3,4]
b=[4,5,6]
a+=b
print(a)# #[1, 2, 3, 4, 4, 5, 6]

# 列表乘以某个数就是列表长度乘以某个数
oa=[0.0]*5
print(oa)#[0.0, 0.0, 0.0, 0.0, 0.0]

# 将字符串类型的数据转换成list类型
index=list("abcd")
print(index) # ['a', 'b', 'c', 'd'] 将其转换成列表


# list中extend和append的不同点
'''
append()和extend()方法都是用来添加数据到list末尾的，两者的区别：
append()添加的时候会把添加的数据当成一个整体进行添加，允许添加任意类型的数据
extend()添加的时候会把添加的数据迭代进行添加，只允许添加可迭代对象数据（可迭代对象： 能用for循环进行迭代的对象就是可迭代对象， 比如：字符串，列表，元祖，字典，集合等等 
'''
one_list = [1, 2, 3, 4, 5]
two_list = ["aaa", "bbb", "ccc"]
one_list.append(two_list)  #  append将其当做整体来看待 
print(one_list)  # [1, 2, 3, 4, 5, ['aaa', 'bbb', 'ccc']]

one_list = [1, 2, 3, 4, 5]
two_list = ["aaa", "bbb", "ccc"]
one_list.extend(two_list)   # 没有将其当成整体来看待
print(one_list)  # [1, 2, 3, 4, 5, 'aaa', 'bbb', 'ccc']


去除list中的None和nan值
list_a=[None,1,1,3]
while None in list_a:
	list_a.remove(None)
```

#### 1.4元祖tuple

- tuple 是使用 ( ) 小括号包含各个数据项
- tuple 与 list 的唯一区别是 tuple 的元素是不能修改，而 list 的元素可以修改
- 元祖使用小括号()或tuple()创建，元素间用逗号分割
- 对元祖类型进行操作时并不改变元祖类型值，而是生成了一个新的元祖

```python
#!/usr/bin/env python3

tuple1 = (True, 1, 'Hello')
print(tuple1)

def func():
    return 1,2 #函数如果有多个返回值，返回的就是元祖类型
print(func())  #输出(1, 2)
```



#### 1.5集合set

- set 是一个**无序**不重复元素的序列 (其无序就导致其不能根据索引来输出相应的内容)
- 使用大括号 { } 或者 set() 函数创建集合
- 用 set() 创建一个空集合,不能用空的{},     因为空的{}大括号返回的是字典
- 使用 set 可以去重
- 集合里面的数据不能是可变数据类型，比如列表就是可变数据类型的数据(里面可以是数字、字符串和元祖)

```python

set1 = {'me', 'you', 'she', 'me'}
print(set1)

a=set()
a.add(1)
a.add(2)
```

1.怎么理解 tuple, list 是有序的序列，而 set 是无序的序列

tuple 或 list 的定义元素顺序和输出一样，而 set 不是。



**set()函数的使用**

```python
#1、最基本用法方法
print(set("python")) #接受一个字符串并且把他们变成一个集合   #输出{'y', 'h', 't', 'n', 'o', 'p'}
print(set([1,2,3]))#接受一个列表并且把他们变成一个集合       #输出{1, 2, 3}
print(set(['i', 'love', 'python']))#这是一个整体了        #输出{'python', 'i', 'love'}

```



#### 1.6字典dictionary

- 字典的每个元素是键值对，**无序**的对象集合
- 字典是可变容器模型，且可存储任意类型对象
- 字典可以通过键来引用，键必须是唯一的且键名必须是不可改变的（即键名必须为Number、String、元组三种类型的某一种），但值则不必
- 字典是使用 { } 大括号包含键值对
- 创建空字典使用 { }

```python

dict1 = {'name': 'steve', 'age': 18}
print(dict1)


dict={1:"wm",2:"wangjun",3:"wangjian"}
print(dict.items()) # dict_items([(1, 'wm'), (2, 'wangjun'), (3, 'wangjian')])
for key,value in dict.items():#遍历
    print(key)
    print(value)
    
>>>#获得里面的值并且是没有任何影响的    
<class 'int'>
<class 'str'>
<class 'int'>
<class 'str'>
<class 'int'>
<class 'str'>

```



### 2、python中的基本用法

#### 1、关于python中的槽

```python
```

#### 2、关于循环

```python
import numpy as np
for i in range(1, 1):  # 没有进行
    print("done once")
# 没有输出任何东西
```









### 3、类的基本概念

通过类定义的数据结构实例。对象包括两个数据成员(类变量和实例变量)和方法

#### 1、python类

1、类提供了一种组合数据和功能的方法。创建一个新类意味着创建一个新的对象类型，从而允许创建一个该类型的新实例

2、每个类的实例可以拥有保存自己状态的属性。一个类的实例也可以有改变自己状态的（定义在类中的）方法

3、Python的类提供了面向对象编程的所有标准特性：
    ⑴类继承机制允许多个基类，派生类可以覆盖它基类的任何方法，一个方法可以调用基类中相同名称的的方法
    ⑵对象可以包含任意数量和类型的数据
    ⑶和模块一样，类也拥有Python天然的动态特性：它们在运行时创建，可以在创建后修改

#### 2、python中类的定义

1、python中定义类使用**class**关键字，class后面紧接类名，类名通常是大写开头的单词(无类继承时类名后可以加括号也可以不加括号)

2、python中类的定义语法如下：

```python
	class ClassName:
		语句1
		...
		语句n
```

注：
1、类定义与函数定义(def语句)一样：只有在被执行才会起作用
    ⑴在定义阶段只是语法检查

2、类是属性和方法的组合，所以语句1可能是内部变量(数据、属性)的定义和赋值语句，也可能是内部方法(函数)的定义语句
    ⑴一个对象的特征称为"属性"
    ⑵一个对象的行为称为"方法"
    ⑶属性在代码层面上来看就是变量，方法实际就是函数，通过调用这些函数来完成某些工作

3、进入类定义时，就会创建一个新的命名空间，并把它用作局部作用域
    ⑴因此，所有对局部变量的赋值都是在这个新命名空间内进行的。特别的，函数定义会绑定到这个局部作用域里的新函数名称

4、正常离开(从结尾出)类定义时，就会创建一个类对象
    ⑴它基本上是一个包围在类定义所创建的命名空间内容周围的包装器
    ⑵元素的(在进入类定义之前起作用的)局部作用域将重新生效，类对象将在这里被绑定到类定义头给出的类名称(在上面的例子中就是ClassName)

```python
class MyClass:
    """定义一个MyClass类"""
    i = 12345

    def func(self):#方法的第一个参数必须为self
        return 'hello world'

```

注：
1、类包含 **属性**和**方法**
    ⑴属性：分为类属性和实例属性
        ①"i = 12345"：表示定义了一个类属性i其值为12345(实例属性后面介绍)
    ⑵方法：即定义在类中的函数(与普通的函数类似)
        ②func：表示定义了一个名为func的实例方法，实际上就是一个稍微特殊点的函数(方法的第一个参数必须为self)

2、在类中定义方法的形式和函数差不多，但其不称为函数，而是叫方法。**方法的调用需要绑定到特定的对象上**(通过self.或实例对象名)，而函数不需要
    ⑴类内部的函数定义通常具有一种特别形式的参数列表，这个特别形式就是第一个参数必须是self(self参数后面介绍)
    ⑵方法是所有实例都共用的：类外所有实例都可以调用类中的方法，类中方法之间也可以相互调用

3、上面例子中创建了一个MyClass抽象类，定义好类后会在当前作用域定义名字MyClass，指向类对象MyClass

4、类也是一种对象类型，跟前面学习过的数值、字符串、列表等等是一样的
    ⑴比如这里构建的类名字叫做MyClass，那么就是我们要试图建立一种对象类型，这种类型被称之为MyClass，就如同有一种对象类型是list一样

5、Python中一切皆对象。不管是我们自己定义的类、6种基本数据类型还是内置的类(第三方类)等都称作为对象
    ⑴一个对象，就会拥有自己的属性和方法。我们可以通过一定的方式来调用一个对象的属性和方法。这一点自定义类与其他Python类型是一样的

6、Python类中的方法分为：实例方法、类方法、静态方法。这里主要介绍实例方法(方法中第一个参数必须为self)，感觉其他两种方法用的比较少

#### 3、类对象

1、定义一个类后，就相当于有了一个类对象了：Python中"一切皆对象"。类也称为"类对象"
    ⑴比如前面例1中定义了类MyClass，其也可以成为类对象

2、类对象支持两种操作：属性引用和实例化
    ⑴实例化：使用instance_name = class_name()的方式实例化，实例化操作创建该类的实例(格式：实例对象名 = 类名()，实例对象名是我们自己定义的)
    ⑵属性引用：使用class_name.attr_name的方式引用类属性(类名.属性名)

**例1：属性引用**

```python

class MyClass:
    """定义一个MyClass类"""
    i = 12345
 
    def func(self):
        return 'hello world'
 
print(MyClass.i) # 引用类属性
#注意下面这个不用加上(),这样的话打印的是函数的信息
print(MyClass.func) # 引用实例方法：实例方法可以这样被引用，但是这样引用无意义(知道即可)
输出：<function MyClass.func at 0x00000295BDBDE3A0> 
    
# 类属性也可以被赋值，因此可以通过赋值来更改类属性的值
MyClass.i = 123
print(MyClass.i)

```

**例2：实例化**

```python
class MyClass:
    """定义一个MyClass类"""
    i = 12345
 
    def func(self):
        return 'hello world'
    
#实例化一个类
my_class = MyClass()
print(my_class)
#输出就是下面这个样子，也就是打印这个类的信息
#<__main__.MyClass object at 0x00000295BDBE7C10>

my_class=MyClass()
print(my_class.i)
#注意这里的函数就要加上()，打印的就是具体的实例化的值
print(my_class.func())
 
#可以看到实例化类后返回的是一个MyClass对象，这个对象跟python中的数字、字符串、列表等是一样的
#对象都可以拥有属性、方法
```

注：
1、类的实例化：是使用函数表示法，可以把类对象看做是会返回一个新的类实例的函数
    ⑴比如上面类对象的实例化就是：my_class = MyClass()。这就创建了一个类的新实例并将此对象分配给局部变量my_class

2、实例化操作可以看成是"调用"类对象：将一个类实例化后获得的对象(所赋值的变量)称为实例对象。my_class就称为实例对象

3、类只是一个抽象的概念，只有经过实例化后(获得实例对象)，才会有意义，才能正常使用这个类中的属性和方法

#### 4、类的实例化

创建类对象的过程又称为类的实例化

实例对象是类对象实例化的产物，实例对象仅支持一个操作：属性引用
   ⑴**实例对象名.属性名**
   ⑵**实例对象名.方法名()**

```python
class MyClass:
    """定义一个MyClass类"""
    i = 12345
 
    def func(self):
        return 'hello world'
 
#第一个实例对象
my_class = MyClass()
print(id(my_class))
print(my_class.i)   # 引用类属性
print(my_class.func()) # 引用实例方法
 
#第二个实例对象
my_class1 = MyClass()
print(id(my_class1))
# 类属性重新赋值
my_class1.i = 123
print(my_class1.i)   # 引用类属性
print(my_class1.func()) # 引用实例方法
 
#第三个实例对象
my_class2 = MyClass()
print(id(my_class2))
print(my_class2.i)   # 引用类属性
print(my_class2.func()) # 引用实例方法
 
"""
2205374276776
12345
hello world
2205374276552
123
hello world
2205374279632
12345
hello world
"""
```

**注：**

1、在未实例化类时(my_class = MyClass()前)，只是定义了类对象的属性和方法，此时其还不是一个完整的对象，将定义的这些称为类(抽象类)。需要使用类来创建一个真正的对象，这个对象就叫做这个类的一个实例，也叫作实例对象(一个类可以有无数个实例)

下面是要搞的另外一个东西

```python
"""这种写法知道就好了，实际中肯定不能这么写：未进行赋值操作！！！！！ 并没有实例化一个对象""" 
class MyClass:
    """定义一个MyClass类"""
    i = 12345
 
    def __init__(self,name):
        self.name = name
 
 
    def func(self):
        print(self)
        print("名字是：%s" % self.name)
 
 
# 未进行赋值操作
print(MyClass("张三").i)   # 引用类属性
print(MyClass("张三").func()) # 引用实例方法
 
print(MyClass("李四").i)   # 引用类属性
print(MyClass("李四").func()) # 引用实例方法
 
"""
12345
<__main__.MyClass object at 0x000001356F83D780>
名字是：张三
None
12345
<__main__.MyClass object at 0x000001356F83D780>
名字是：李四
None
"""
```

1、如果在实例化类时，未将实例赋值给一个变量：虽然可以正常调用类的属性和方法
    ⑴但是这样是没有意义的。因为：没有任何引用指向这个实例，都没法调用这个实例(只有赋值后才会产生实例对象)

2、如果这样写的话，每次调用这个类的实例对象都需要去实例化一次了，那么就显得很麻烦了，还不如实例化一次并赋值给一个变量，此后每次去调用这个变量(实例对象)就好了

3、因此：类在使用前必须先实例化，并将实例赋值给一个变量(得到实例对象)



#### 5、self参数

1、在定义实例变量、实例方法时的第一个参数必须是self
    ⑴其实：self名称不是必须的，在python中self不是关键词，你可以定义成a或b或其它名字都可以，只是约定成俗都使用了self
    ⑵也就是说在定义实例方法时必须有一个参数是默认已经存在了的，可以是self，可以是a，也可以是b。不管这个参数名是什么，但必须得有这个参数
2、**self在定义时需要定义，但是在调用时会自动传入(不需要手动传入了)**

3、self其实就相当于C++中的this指针



```python
class MyClass:
    """定义一个MyClass类"""
    i = 12345
 
    def func(self):
        print("self参数：",self)
        return 'hello world'
 
a = MyClass()
print("实例对象：",a)
a.func()
 
b = MyClass()
print("实例对象：",b)
b.func()
 
"""
实例对象： <__main__.MyClass object at 0x000002D29354E3C8>
self参数： <__main__.MyClass object at 0x000002D29354E3C8>

实例对象： <__main__.MyClass object at 0x000002D29354EFD0>
self参数： <__main__.MyClass object at 0x000002D29354EFD0>
"""
```

1、通过打印的id值可以看出，self参数实际上就是类通过实例化后得到的实例对象。不同的实例对象对应的self参数是不一样的(self参数始终与当前实例对象时一一对应的)

2、在这个例子可中可能并不能很好的理解self参数的含义，感觉是实例变量中能更好的理解self参数

3、目前我们只需记住：
    ⑴实例方法第一个参数必须是self，在调用时会自动传入(不需要手动传入了)
    ⑵self代表的当前的实例对象本身



```python
class Ball:
    def setname(self,name,age):
        self.name = name
        print(age)
 
    def kick(self):
        return "我叫%s" % self.name
 
a = Ball()
b = Ball()
c = Ball()
 
a.setname("A",1)
b.setname("B",2)
c.setname("C",3)
 
print(a.kick())
print(b.kick())
print(c.kick())
 
"""
1
2
3
我叫A
我叫B
我叫C
"""
```

在方法中定义的参数，一般来说只能在当前方法中使用(作用域)
    ⑴如果想要一个方法中的参数能在其他方法中使用，那么就可以使用"self."来将这个参数变成一个实例变量(实例变量后面介绍，这里主要是遇到了这种写法)
    ⑵name参数：在方法中使用了"self.name = name"，这步就相当于是将这个name参数变成了一个实例变量，因此可以在所有方法中使用(这种写法了解即可，没啥意义，因为一个实例变量最好直接定义在__init__方法中)
    ⑶age参数：age参数就没有使用name参数那样的写法，仅仅是在setname()方法中定义并使用，因此age参数就只能在setname()方法中使用，而不能在kick()方法中使用，即使他们是在同一个类中(经常遇到的是这种写法)



**类变量(类属性)**:所有的变量共有的，在类中定义的，每一个类的实例化对象都一样

**实例变量**:每一个变量独有的，定义在__init__方法中的，根据初始化传入的东西不同而不同



#### 6、类变量

1、类变量：是该类所有实例对象共享的属性(也可以叫"类属性")
    ⑴类属性是所有实例都共用的：所有实例都可以调用这个类属性
    ⑵在类中任意地方(所有方法中)都可以使用"类名.类属性名"来调用类属性
    ⑶在类外任意地方都可以使用"类名.类属性名"或"实例名.类属性名"来调用类属性

 2、类变量是直接定义在类中的，比如例1中的"i = 12345"，变量i就是一个类属性，该变量是所有实例对象共有的。类中的所有方法、实例都可以使用它

```c++
class Car():
    """这是一个汽车类"""
    brand = "宝马"
 
    def run(self, s):
        # 类中调用类属性：类名.属性名
        print("当前车型为：%s,当前行驶速度：%s KM/S" % (Car.brand,s))
 
 
a = Car()
# 类外调用类属性：实例名.属性名
print(a.brand, id(a.brand))
a.run(110)
 
b = Car()
print(b.brand, id(b.brand))
b.run(200)
 
"""
宝马 1744629351728
当前车型为：宝马,当前行驶速度：110 KM/S
宝马 1744629351728
当前车型为：宝马,当前行驶速度：200 KM/S
"""
```



#### 7、实例变量

1、类变量是所有实例公用的属性。也就是说一些属性是所有实例都共有的，那么此时我们可以将该属性定义为类属性
    ⑴那么如果某些属性是每个实例独有的(每个实例的属性值都不一致)，那么我们就可以将这些属性定义为实例属性

2、实例变量：是每个实例都独有的数据(也可以叫"实例属性")
    ⑴即某个属性对于每个实例都是独有的，就需要将其定义为实例变量
    ⑵某个属性是每个实例同共有的就可以定义为类属性

3、实例变量是定义在__init__方法中的
    ⑴__init__()方法也是类中的一个方法，因此其第一个参数也必须是self
    ⑵实例变量是每个实例对象独有的，因此在定义实例变量时，必须是：self.实例变量名 = 外部形参名(通过self来绑定当前实例对象) 
    ⑶在类中任意地方(所有方法中)都可以使用"self.实例属性名"来调用实例属性
    ⑷在类外任意地方都可以使用"实例名.实例属性名"来调用实例属性

```python
class People():
    country = "china"
 
    def __init__(self,name):
        self.name = name
 
    def speak(self, age):
        # 类中调用类属性：类名.属性名
        # 类中调用实例属性：self.属性名
        print("我的名字是：%s,来自：%s，年龄是：%s" % (self.name,People.country,age))
        #这个地方People也可以换成self
 
# 实例化类时传入实例变量值
a = People("Tom")
a.speak(11)
 
b = People("Jack")
b.speak(12)
 
"""
我的名字是：Tom,来自：china，年龄是：11
我的名字是：Jack,来自：china，年龄是：12
"""
```

因此**在类中调用实例属性时都必须使用"self.实例属性名"的方式来调用**，这样才能通过self参数来确定当前是哪个实例对象在调用



#### 8、类变量与实例变量

在调用一个类中的属性时，Python会按照一定的顺序去查找这个属性：**先在当前实例中找，有就用当前实例中的，如果没有就找类中的**

```python
class C:
    count = 0
a = C()
b = C()
c = C()
print(a.count,b.count,c.count)   #output:0,0,0
 
a.count += 10   #实例对象调用类属性
print(a.count,b.count,c.count)   #output:10,0,0
 
C.count += 100  #类对象调用类属性
print(a.count,b.count,c.count)   #output:10 100 100
 
#print(count)   #name 'count' is not defined，不能直接访问类属性，具体访问方法参考前面的属性访问
```



注：

1、对实例对象的count属性进行赋值后，就相当于覆盖了类对象C的count属性，如果没有赋值覆盖，那么引用的就是类对象的count属性
    ⑴通过"实例对象名.属性名"来覆盖类属性，只会影响到当前实例对象，不会影响到其他实例对象中的类属性
    ⑵通过"类名.属性名"来覆盖类属性，会影响到所有实例的类属性
    ⑶因此在类外调用类变量时，最好使用"实例对象名.属性名"，避免在重新赋值时影响到其他实例

2、类变量和实例变量的区别在于：类变量是所有对象共有，其中一个对象将它值改变，其他对象得到的就是改变后的结果；而实例变量则属对象私有，某一个对象将其值改变，不影响其他对象

3、获取一个实例对象的属性时，其属性名后面都是不需要加园括号的(不管是我们自己定义的类还是Python自带的类)；如果属性名后面带上了园括号，那么就变成了一个方法名了。这肯定是不对的。所以要分清楚调用的是类的属性还是类的方法
    ⑴不管是调用类方法还是属性，都是通过"."点操作来实现的



#### 9、类的权限

```python
```









### 4、类的扩展概念

1、定义个类主要是将一些具有相同属性的数据、方法放到一个类中整合起来方便代码的管理。最终目的还是调用类中的属性和方法

2、在调用类中的方法或属性时都必须遵循一定的规则：调用类的属性或方法分类在类中调用、在类外调用。不同地方调用，调用的方式也会有一定的差距

3、调用类属性：
    ⑴类中访问类变量：类名. 类变量名
    ⑵类外访问类变量：类名.类变量名或实例名.类变量名

4、调用实例属性：
    ⑴类中访问实例变量：self.实例变量名
    ⑵类外访问实例变量：实例名.实例变量名

4、调用实例方法：
    ⑴类中访问实例方法：self.方法名(参数)或类名.方法名(self,参数)
    ⑵类外访问实例方法：实例名.方法名(参数)

#### 1、类的使用

```python
class MyClass:
    """一个简单的类实例"""
    i = 12345#定义一个类属性
 
    def f(self):#定义一个实例方法
        print(MyClass.i) #类中调用类属性
        return 'hello world'
 
    def g(self):
        # 类中调用实例方法
        self.f()
        MyClass.f(self)
 
x = MyClass()   # 实例化类
 
# 访问类的属性和方法
print("MyClass 类的属性 i 为：", x.i) #类外调用类属性
print("MyClass 类的属性 i 为：", MyClass.i)
print("MyClass 类的方法 f 输出为：", x.f()) #类外调用实例方法
x.g()
 
"""
MyClass 类的属性 i 为： 12345
MyClass 类的属性 i 为： 12345
12345
MyClass 类的方法 f 输出为： hello world
12345
12345
"""

```

注：
1、在未实例化类时(x = MyClass()前)，只是定义了对象的属性和方法，此时其还不是一个完整的对象，将定义的这些称为类(抽象类)。需要使用类来创建一个真正的对象，这个对象就叫做这个类的一个实例，也叫作实例对象(一个类可以有无数个实例)

2、创建一个对象也叫做类的实例化，即x = MyClass()。(此时得到的x变量称为类的具体对象)。注意此时类名后面是跟着小括号的，这跟调用函数一样。另外赋值操作并不是必须的，但如果没有将创建好的实例对象赋值给一个变量，那这个对象就没办法使用，因为没有任何引用指向这个实例

3、如果要调用对象里的方法，就需要判断是在类中调用还是在类外调用：
    ⑴在类中调用实例方法：self.方法名(参数)或类名.方法名(self,参数)这个这里只是提一下，可以先不纠结
    ⑵在类外调用实例方法：实例对象名.方法名(参数)。这里的例子就是类外调用方法，只是说实例方法中没有定义参数。x.f()：x为实例对象名，f()为类中定义的实例方法
    
4、 x.i和MyClass.i都是用于调用类的属性，也就是我们前面所说的类变量；x.f()用于调用类的实例方法
    ⑴调用类属性可以使用：实例对象名.类属性名或类名.类属性名。虽然这两种方法都可以调用类属性，但是两者在调用使用还是有区别的，后面介绍

 5、类中定义方法的要求：在类中定义方法时，第一个参数必须是self，除第一个参数外，类的方法和普通的函数没什么区别，如可以使用默认参数，可变参数，关键字参数和命名关键字参数等
    ⑴虽然在定义方法时会定义一个self参数，但是不管是在类中或是在类外调用方法都不用传递self，其他参数正常传入
    ⑵self参数究竟是什么，这里也可以先不纠结，目前只需要知道定义一个实例方法时，第一个参数必须是self，但是在调用实例方法时，不需要传递这个self参数

6、类对象(抽象类)支持两种操作：即属性引用和实例化
    ⑴**属性引用**：方法为类名.类属性名(也可以实例对象名.类属性名)
    ⑵**实例化**：将一个抽象类实例化成一个实例对象(x = MyClass() )。一个类可以实例化出无数个对象

7、类是一个抽象的概念，对象则是一个实际存在的东西。就像我们说的"狮子"，它只是一个抽象的东西，只有具体到狮子这种动物身上它才是实际存在的。在比如设计房子的图纸只能告诉你房子是什么样的，并不是真正的房子，只有通过钢筋水泥建造出来的房子才实际存在，才能住人。
    ⑴"造房子"这个过程就相当于是"实例化类'，一个抽象类只有实例化成一个具体的实例对象后，才会有意义(抽象类只有实例化后才能使用，才会有意义)



**例8：**

```python
class Student():
    address = "china"     #定义类变量address
 
    def __init__(self,name,age):    #定义实例变量age和name
        self.name = name
        self.age = age
 
    def Info(self,score):#定义在方法中的变量(普通的变量：作用域为这个方法内)
        return "学生来自于%s,名字为%s,年龄为%s,成绩为%s"%(Student.address,self.name,self.age,score)
        #类中访问实例变量：self.实例变量名
        #类中访问类变量：类名.类变量名
        #类中访问方法中的普通变量：直接变量名(且该变量只能在这个方法中使用，不能再其他方法或类外调用)
 
 
student = Student("张三",18)  #实例化类
print(student.name) #类外访问实例变量：实例名.实例属性名
print(Student.address)  #类外访问类变量：类名.类属性名(也可以实例名.类属性名)
print(student.Info(98)) #类外访问实例方法：实例名.方法名(参数)
 
#另一个实例对象
student_1 = Student("李四",20)
print(student_1.name)
print(student_1.address)
print(student_1.Info(100))
 
"""
张三
china
学生来自于china,名字为张三,年龄为18,成绩为98
李四
china
学生来自于china,名字为李四,年龄为20,成绩为100
"""
```

注：
1、在Student类中，类属性address为所有实例所共享；实例属性name和age每个student的实例独有(每个实例有不同的name和age)
    ⑴类属性：实例对象student和student_1拥有一样的address属性
    ⑵实例属性：实例对象student和student_1拥有不一样的name和age属性(每个实例独有的属性)

#### 2、属性绑定

1、在定义类时，通常我们说的定义属性，其实是分为两个方面的：类属性绑定、实例属性绑定(也就是定义类属性或实例属性)

2、用绑定这个词更加确切；不管是类对象还是实例对象，属性都是依托对象而存在的。我们说的属性绑定，首先需要一个可变对象，才能执行绑定操作，使用objname.attr = attr_value的方式，为对象objname绑定属性attr。这分两种情况：
    ⑴若属性attr已经存在，绑定操作会将属性名指向新的对象
    ⑵若不存在，则为该对象添加新的属性，后面就可以引用新增属性

##### 1、类属性绑定

Python作为动态语言，类对象和实例对象都可以在运行时绑定任意属性。因此，类属性的绑定发生在两个地方
   ⑴类定义时
   ⑵运行时任意阶段

```python
class Dog:
 
    kind = 'canine'
 
Dog.country = 'China' #绑定一个新的类属性country
 
print(Dog.kind, ' - ', Dog.country)  # 输出: canine  -  China
del Dog.kind
print(Dog.kind, ' - ', Dog.country)  #由于上一行删除的kind属性，因此输出为AttributeError: type object 'Dog' has no attribute 'kind'
```

**注：**
1、在类定义中，类属性的绑定并没有使用objname.attr = attr_value的方式，这是一个特例，其实是等同于后面使用类名绑定属性的方式

2、因为是动态语言，所以可以在运行时增加属性，删除属性



##### 2、实例属性绑定

与类属性绑定相同，实例属性绑定也发生在两个地方：**类定义**时、**运行时任意阶段**

```python
class Dog:
 
    def __init__(self, name, age):
        self.name = name
        self.age = age
 
dog = Dog('Lily', 3)
dog.fur_color = 'red' #为实例对象dog增加一个fur_color属性
 
print('%s is %s years old, it has %s fur' % (dog.name, dog.age, dog.fur_color))
 
#上面代码的输出结果为：Lily is 3 years old, it has red fur

'''
cat=Cat("wangming",22)
print('%s is %s years old, it has %s fur' % (cat.name, cat.age, cat.fur_color))
'''
#输出上面这句话，最后会报错，也就是实例属性绑定的结果只有这个实例拥有
```

注：
1、语句self.name = name，self.age = age以及后面的语句dog.fur_color = 'red'为实例dog增加三个属性name, age, fur_color。

2、Python类实例有两个特殊之处：
    ⑴__init__在实例化时执行
    ⑵Python实例对象调用方法时，会将实例对象作为第一个参数传递因此，__init__方法中的self就是实例对象本身，这里是dog



#### 3、属性引用

##### 1、类属性引用

属性的引用与直接访问名字不同，不涉及到作用域

```python
class Dog:
    kind = 'canine'
 
    def tell_kind(self):
        print(Dog.kind)
 
    def info(self):
        return self.tell_kind() #类中调用实例方法
 
dog = Dog()
dog.tell_kind()  # Output: canine
dog.info()  # Output: canine        注意这里返回的时候也实现了他
```

##### 2、实例属性引用

使用实例对象引用属性稍微复杂一些，因为实例对象可引用类属性以及实例属性。但是**实例对象引用属性**时遵循以下规则：
   ⑴总是**先到实例对象中查找属性，再到类属性中查找属性**
   ⑵属性绑定语句总是为实例对象创建新属性，属性存在时，更新属性指向的对象



```python
class Dog:
 
    kind = 'canine'
    country = 'China'
 
    def __init__(self, name, age, country):
        self.name = name
        self.age = age
        self.country = country
 
dog = Dog('Lily', 3, 'Britain')
print(dog.name, dog.age, dog.kind, dog.country)   #output:Lily 3 canine Britain
```

**注：**
类对象Dog与实例对象dog均有属性country，按照规则，dog.country会引用到实例对象的属性；但实例对象dog没有属性kind，按照规则会引用类对象的属性



```python
class Dog:
 
    kind = 'canine'
    country = 'China'
 
    def __init__(self, name, age, country):
        self.name = name
        self.age = age
        self.country = country
 
dog = Dog('Lily', 3, 'Britain')
 
print(dog.name, dog.age, dog.kind, dog.country)   # Lily 3 canine Britain
print(dog.__dict__)                               # {'name': 'Lily', 'age': 3, 'country': 'Britain'}
 
dog.kind = 'feline'
 
print(dog.name, dog.age, dog.kind, dog.country)   # Lily 3 feline Britain
print(dog.__dict__)                               # {'name': 'Lily', 'age': 3, 'country': 'Britain', 'kind': 'feline'}
print(Dog.kind)                                   # canine (没有改变类属性的指向)
```

**注：**
1、使用属性绑定语句dog.kind = 'feline'，按照规则，为实例对象dog增加了属性kind，后面使用dog.kind引用到实例对象的属性。这里不要以为会改变类属性Dog.kind的指向，实则是为实例对象新增属性，可以使用查看__dict__的方式证明这一点。

**dog.__ dict __** 会输出所有的属性

**Dog.__ dict __**会把所有的属性和行为以字典的形式输出



##### 3、可变属性引用

```python
class Dog:
    
    tricks = []   #这个是类属性，所以实例化对象操作类属性的时候是他们共有的
 
    def __init__(self, name):
        self.name = name
 
    def add_trick(self, trick):
        self.tricks.append(trick)
 
d = Dog('Fido')s
e = Dog('Buddy')
d.add_trick('roll over')
e.add_trick('play dead')
print(d.tricks)             output:# ['roll over', 'play dead']
```



#### 4、实例方法、类方法和静态方法

在面向对象的编程中，类属性可细分为类属性和实例属性一样，同样的，对于类中的方法也可以具体可划分为类方法、实例方法和静态方法



##### 1、实例方法

1、在类编程中，一般情况下在类中定义的方法、函数默认都是实例方法

2、python的类编程中实例方法最大的特点就是最少要包含一个self参数，该参数必须定义，但调用时不需要传
    ⑴该self参数的作用是绑定调用此方法的实例对象(确定当前是哪个实例对象在调用方法、实例变量，Python会自动完成绑定)，类比C++中的this指针
    ⑵实例方法：方法中第一个参数都是self,实例变量、实例方法都需要要绑定self(self参数表示当前实例对象本身)
    ⑶调用实例方法：只能由实例对象调用

3、感觉经常用到的都是实例方法，所以前面主要介绍了实例方法。类方法和静态方法感觉不是经常用到，所以在这里补充下

```python
class MyClass:
    className = "三年2班"  # 定义一个类属性className
 
    def __init__(self,name):#定义一个实例变量name
        self.name = name
 
    def BaseInfo(self):
        # 类中调用实例变量、类变量
        baseInfo = "My name is %s,I am a student in %s" % (self.name,MyClass.className)
        return baseInfo
 
    def ComeFrom(self,country,*args): # 在方法中定义一些局部变量(只能在该方法中使用)
        # 类中调用类方法
        baseInfo = self.BaseInfo()
        comeFrom = baseInfo + ".I comefrom " + country + "," + ",".join(args)
        return comeFrom
 
x = MyClass("张三")  # 实例化类
# 类外访问实例方法
print(x.ComeFrom("china","chengdu","高新区","茂业中心"))
 
y = MyClass("李四")  # 实例化类
# 类外访问实例方法
print(y .ComeFrom("china","meishan","hongya","gaomiao"))
 
"""
My name is 张三,I am a student in 三年2班.I comefrom china,chengdu,高新区,茂业中心
My name is 李四,I am a student in 三年2班.I comefrom china,meishan,hongya,gaomiao
"""
```

##### 2、类方法

1、Python中的类方法和实例方法类似，但类方法需要满足以下要求：
    ⑴类方法至少需要包含一个参数，与实例方法不同的是该参数并非self，而是python程序员约定俗成的参数：cls(cls表示当前类对象)
    ⑵Python会自动将类本身绑定到cls参数(非实例对象)，故在调用类方法时，无需显式为cls参数传递参数
    ⑶类方法需要使用修饰语句： ＠classmethod

2、调用类方法：类和实例对象都可以调用
    ⑴类方法推荐使用类名直接调用，当然也可以使用实例对象来调用(不推荐



```python
class CLanguage:
    #类构造方法，也属于实例方法
    def __init__(self):
        self.name = "C语言中文网"
        self.add = "http://c.biancheng.net"
    #下面定义了一个类方法
    @classmethod
    def info(cls):
        print("正在调用类方法",cls)
 
 
#使用类名直接调用类方法
CLanguage.info()
#使用类对象调用类方法
clang = CLanguage()
clang.info()
 
"""
正在调用类方法 <class '__main__.CLanguage'>
正在调用类方法 <class '__main__.CLanguage'>
"""
```



```python
# 文件在工程中的路径：Py_Project/zxc.py
class People():
    def __init__(self,name):
        self.name = name
 
    # 定义了一个类方法：类方法中是不能有实例变量的
    @classmethod
    def Age(self,age):
        age = "age is %s" % (age)
        return age
    # 定义一个实例方法：可正常使用实例变量
    def Info(self,age):
        info = "name is %s,age is %s" % (self.name,age)
        return info
 
 
# 导入所需模块
from Py_Project.zxc import People
 
#调用实例方法：在调用实例方法前必须实例化类
people = People("jack")
print(people.Info(12))
 
# name is jack,age is 12
 
 
# 导入所需模块
from Py_Project.zxc import People
 
#调用类方法:通过实例名来调用类方法，也要先实例化类
people = People("jack")
print(people.Age(13))
 
# 调用类方法:通过类名来调用类方法，就不需要实例化类
print(People.Age(14))
 
# age is 13
# age is 14
```

**注：**

1、在一个类中可以同时定义实例方法、类方法、静态方法

2、类方法中是不能调用实例变量的，但是可以调用类变量：因为类方法是指向类的，而不是实例对象的

3、调用实例方法前必须实例化类；调用类方法就可以直接使用类名进行调用





##### 3、静态方法

1、类中的静态方法，实际上就是大家众所周知的普通函数，存在的唯一区别是：
    ⑴类静态方法在类命名空间中定义，而函数则在程序的全局命名空间中定义

2、需要注意的是：
    ⑴类静态方法没有self、cls这样的特殊参数，故Python解释器不会对其包含的参数做任何类或对象的绑定
    ⑵类静态方法中无法调用任何类和对象的属性和方法，类静态方法与类的关系不大
    ⑶静态方法需要使用**＠staticmethod**修饰

3、静态方法的调用，既可以使用类名，也可以使用类对象

4、静态方法是类中的函数，不需要实例等
    ⑴静态方法主要是用来存放逻辑性的代码，逻辑上属于类，但是和类本身没有关系
    ⑵也就是说在静态方法中，不会涉及到类中的属性和方法的操作**(静态方法中不能使用实例变量、类变量、实例方法等)**
    ⑶可以理解为，静态方法是个独立的、单纯的函数，它仅仅托管于某个类的名称空间中，便于使用和维护



```python
class CLanguage:
    @staticmethod
    def info(name,add):
        print(name,add)
#使用类名直接调用静态方法
CLanguage.info("C语言中文网","http://c.biancheng.net")
 
#使用类对象调用静态方法
clang = CLanguage()
clang.info("Python教程","http://c.biancheng.net/python")
 
"""
C语言中文网 http://c.biancheng.net
Python教程 http://c.biancheng.net/python
"""
```







### 4、类的其他

#### 1、例1

```python
class CC:
    def setXY(self, x, y):
        self.x = x
        self.y = y

    def printXY(self):
        print(self.x, self.y)


dd = CC()
dd.setXY(4, 5)
dd.printXY()#输出4 5
```

#### 2、例2

```python
class TestClass(object):
    val1 = 100

    def __init__(self):
        self.val2 = 200

    def fcn(self, val=400):
        val3 = 300

        self.val4 = val
        self.val5 = 500


inst = TestClass()

print(TestClass.val1) #val1是类属性
print(inst.val1) #通过实例对象来调用类的属性
print(inst.val2)#val2是实例属性
#inst.fcn() 增加这一句话之后，val4和val5就是实例变量了
print(inst.val4)
print(inst.val5)
```

1、val1是类变量，可以由类名直接调用，也可以有对象来调用；

2、val2是实例变量，可以由类的对象来调用，这里可以看出成员变量一定是以self.的形式给出的，因为self的含义就是代表实例对象

3、val3既不是类变量也不是实例变量，它只是函数fcn内部的局部变量

4、val4和val5也都不是实例变量，虽是以self.给出，但并没有在构造函数中初始化



#### 3、抽象类

在python中类 通过继承`metaclass =ABCmeta`类来创建抽象类，抽象类是包含抽象方法的类，其中ABCmeta类`（Metaclass for defining abstact baseclasses,抽象基类的元类）`是所有抽象类的

定义了抽象类后，在要实现抽象方法的前一行使用`@abc.abstractmethod`来定义抽象方法。抽象方法不包含任何可实现的代码，只能在子类中实现抽象函数的代码。



**抽象类的特点**

- 抽象类不能被实例化
- 继承抽象类的类必须实现所有的抽象方法后，才能被实例化
- 抽象类中有抽象方法和正常方法
- 抽象类中可以添加成员属性

```python
import abc

class ChouxiangleiA(metaclass=abc.ABCMeta):  # 这里的ChouxiangleiA就是抽象类，必须指定abc模块中的元类ABCMeta
    @abc.abstractmethod  # 定义抽象方法，这里的写法是固定的，
    def abstract_method(self):
        """子类中必须定个抽象类中的方法，这里固定用pass"""
        pass
 
 
 
class B(ChouxiangleiA):  # B作为抽象类ChouxiangleiA的子类，必须定义抽象方法
    def abstract_method(self):
        print('抽象方法')
 
b = B()
b.abstract_method() # 抽象方法
```

**注意**：

- 抽象类中的抽象方法，在子类中必须实现该方法
- 抽象类不能被实例化
- 创建抽象方法前，必须先创建抽象类
- 需要导入abc（abstractclass）类



**抽象类的使用**

- 直接继承

直接继承抽象基类的子类就没有这么灵活，抽象基类中可以声明”抽象方法“和“抽象属性”，**只有完全`重写`（实现）了抽象基类中的“抽象”内容后，才能被实例化**，而虚拟子类则不受此影响。

- 虚拟子类

将其他的类”注册“到抽象基类下当虚拟子类（调用register方法），虚拟子类的好处是你实现的第三方子类不需要直接继承自基类，可以实现抽象基类中的部分API接口，也可以根本不实现，但是issubclass(), issubinstance()进行判断时仍然返回真值。

```python
import abc
class Animal(metaclass=abc.ABCMeta): # 继承自metaclass=abc.ABCMeta，表明这是一个抽象类
    @abc.abstractmethod  # 被abc.abstractmethod所装饰的方法为一个抽象方法
    def eat(self):
        pass

    @abc.abstractmethod  
    def run(self):  # 这格式一个抽象方法
        pass

class Dog(Animal):
    """
    继承Animal这个抽象类，则需要实现这个抽象类中的所有抽象方法，否则无法实例化
    """
    def eat(self):
        print("dog is eating")

    def run(self):
        print("dog is running")


class Duck(Animal):
    """
    继承Animal这个抽象类，则需要实现这个抽象类中的所有抽象方法，否则无法实例化
    """
    def eat(self):
        print("duck is eating")

    # def run(self):
    #     print("duck is running")


dog = Dog()
duck = Duck()#Duck类不能被实例化
dog.eat()
dog.run()
duck.eat() # 该方法重写了，但是抽象父类中的run（）抽象方法没有被重写，则该类无法实例化


#结果 
Traceback (most recent call last):
  File "D:/py3.7Project/sample/chapter01/all_is_object.py", line 80, in <module>
    duck = Duck()
TypeError: Can't instantiate abstract class Duck with abstract methods run


```





### 5、Python中 *args，**args的详细用法

`*args` 和 `**kwargs`主要用于函数定义，你可以将不定数量的[参数传递](https://so.csdn.net/so/search?q=参数传递&spm=1001.2101.3001.7020)给某个函数



1、*args

*args 不定参数（不定的意思是指，预先并不知道，函数使用者会传递多少个参数给你）

*args是用来发送一个非键值对的可变数量的参数列表给一个函数。
*args的用法：当传入的参数个数未知，且不需要知道参数名称时。

```python
def func_arg(farg, *args):
    print("formal arg:", farg)
    for arg in args:
        print("another arg:", arg)
func_arg(1,"youzan",'dba','hello')



# formal arg: 1
# another arg: youzan
# another arg: dba
# another arg: hello

```



```python
b=(1,2,3)
print(*b)
<<<1 2 3 #加上*就是将这个元祖给分解了


def nihao(a,b,c):
    print("a=",a," b=",b," c=",c)
    
nihao(*b)
>>>a= 1  b= 2  c= 3
#将上面直接给分解了
```









2、**args

** kwargs 传入键值对(例如：num1=11,num2=22)

** kwargs 允许将不定长度的键值对作为参数传递给一个函数。如果想要在一个函数里处理带名字的参数，应该使用 **kwargs。



```python
#利用它转换参数为字典
def kw_dict(**kwargs):
    return kwargs
print(kw_dict(a=1,b=2,c=3))


# {'a': 1, 'b': 2, 'c': 3}

```

### 6、python中字符串格式化的方法

#### 1、使用.format()方法

 该`format`方法是在Python 2.6中引入的，是字符串类型的内置方法。因为str.format的方式在性能和使用的灵活性上都比%号更胜一筹，所以推荐使用

**1、使用位置参数**

```python
# 按照位置一一对应
print('{} asked {} to do something'.format('egon', 'lili'))  # egon asked lili to do something
print('{} asked {} to do something'.format('lili', 'egon'))  # lili asked egon to do something
```

**2、使用索引**

```python
# 使用索引取对应位置的值
print('{0}{0}{1}{0}'.format('x','y')) # xxyx
```

**3、使用关键字参数or字典**

```python
可以通过关键字or字典方式的方式格式化，打破了位置带来的限制与困扰
print('我的名字是 {name}, 我的年龄是 {age}.'.format(age=18, name='egon'))

kwargs = {'name': 'egon', 'age': 18}
print('我的名字是 {name}, 我的年龄是 {age}.'.format(**kwargs)) # 使用**进行解包操作

```

**4、填充与格式化**

```python
# 先取到值,然后在冒号后设定填充格式：[填充字符][对齐方式][宽度]   注意这个里面的填充字符不需要加单引号双引号或者是其他的表示它是字符的东西

# *<10：左对齐，总共10个字符，不够的用*号填充
print('{0:*<10}'.format('开始执行')) # 开始执行******

# *>10：右对齐，总共10个字符，不够的用*号填充
print('{0:*>10}'.format('开始执行')) # ******开始执行

# *^10：居中显示，总共10个字符，不够的用*号填充
print('{0:*^10}'.format('开始执行')) # ***开始执行***

```

**5、精度与进制**

```python
print('{salary:.3f}'.format(salary=1232132.12351))  #精确到小数点后3位，四舍五入，结果为：1232132.124
print('{0:b}'.format(123))  # 转成二进制，结果为：1111011
print('{0:o}'.format(9))  # 转成八进制，结果为：11
print('{0:x}'.format(15))  # 转成十六进制，结果为：f
print('{0:,}'.format(99812939393931))  # 千分位格式化，结果为：99,812,939,393,931

```



#### 2、使用格式化字符串 %（参数1，参数2......）

一般情况下，不采用这种方法

```python
# 1、格式的字符串（即%s）与被格式化的字符串（即传入的值）必须按照位置一一对应
# ps：当需格式化的字符串过多时，位置极容易搞混
print('%s asked %s to do something' % ('egon', 'lili'))  # egon asked lili to do something
print('%s asked %s to do something' % ('lili', 'egon'))  # lili asked egon to do something

# 2、可以通过字典方式格式化，打破了位置带来的限制与困扰
print('我的名字是 %(name)s, 我的年龄是 %(age)s.' % {'name': 'egon', 'age': 18})

kwargs={'name': 'egon', 'age': 18}
print('我的名字是 %(name)s, 我的年龄是 %(age)s.' % kwargs)

```



### 7、dir()函数和help()

```python
#help函数查看当前和这个对象的说明书
help(torch.cuda.is_available)#这里注意不要加上torch.cuda.is_available()
#dir函数查看每一个模块都在哪一个地方
dir(torch)
dir(torch.cuda)
```

### 8、类名+??

```python
#直接用类名+??就可以直接显示这个类的详细信息
Dataset??
```

### 9、Dataset和Dataloader

```python
#Dataset,是提供一种方式去获取数据及其label
1、如何获取每个数据及其label
2、告诉我们总共有多少数据


#Dataloader,为网络提供不同的数据形式
```



### 10、windows下路径问题

```python
/(除号)是正斜杠,linux系统的路径
\是反斜杠，windows系统用的

Window下python读取数据路径可以有三种表示方式： 
(1)'c:\\a.txt' #转义的方式。表示这里\\是一个普通\字符，不容易出错
(2)r'c:\a.txt' #声明字符串。声明字符串，表示不需要转义，这里\就是一个普通字符串
(3)'c:/a.txt'  #直接使用正斜杠表示路径。与linux系统一样，没有转义的误解


在Unix/Linux中，路径的分隔采用正斜杠"/"，比如"/home/hutaow"；

而在Windows中，正反斜杠二者皆可表示路径，通常看到是路径分隔采用反斜 杠"\"，比如"C:\Windows\System"。
有时我们会看到这样的路径写法，"C:\\Windows\\System"，也就是用两个反斜杠来分隔路径，这种写法在网络应用或编程中经 常看到，事实上，上面这个路径可以用"C:/Windows/System"来代替，不会出错。但是如果写成了"C:\Windows\System"， 那就可能会出现各种奇怪的错误了。
```



### 11、with是如何工作的

**基本逻辑**:with语句with 语句所求值(也就是上下文管理器)的对象必须有一个 __enter__() 方法和一个 __exit__() 方法。

当紧跟着with后面的语句被求值后，这类的__ enter __方法被调用，这个方法的返回值将被赋值给as后面的变量。

当with后面的代码块全部执行完之后，将调用前面返回对象的——exit——方法。

```python
class Simple:
    def __init__(self,name,age):
        self.name=name
        self.age=age
        print("name:",name)
        print("age:",age)


    def __enter__(self):
        print("In __enter__()")
        return "enter"

    def __exit__(self, type, value, trace):
        print("In __exit__()")

def get_simple():
    return Simple()

with Simple("wangming",22) as simple:
    print("simple:", simple)
    
'''
先执行Simple("wangming",22)及其中的init，然后执行enter并且将返回值赋给simple,然后执行simple后面的代码块，最后执行前面的exit方法
'''
    
运行结果：    
name: wangming
age: 22
In __enter__()
simple: enter
In __exit__()


```

没有as的情况也可以，as后面的对象其实是__ enter __函数的返回值，如果没有返回值的话，也可以不用as

```python
class Hand:
    def __enter__(self):
        print("进入__enter__方法")
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("进入__exit__方法")

with Hand():
    print("gett..")
#这里__enter__没有返回值，所以可以不用as
```

在exit函数里面存在这样几个参数type， value， trace

```python
class Hand:
    def __enter__(self):
        print("进入__enter__方法")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("进入__exit__方法")
        print("type:",exc_type)
        print("value:",exc_val)
        print("trace:", exc_tb)
    def cal(self):
        return 100/0 #这个是一个不合法的语句

with Hand() as h:
    print("gett..")
    #h.cal()

#输出1：没有h.cal()
进入__enter__方法
gett..
进入__exit__方法
type: None
value: None
trace: None
    
#输出2：存在h.cal()
进入__enter__方法
gett..
进入__exit__方法
type: <class 'ZeroDivisionError'>
value: division by zero
trace: <traceback object at 0x000002840DA8BA40>
#也就是退出函数里面的那些实参保存的就是类里面成员函数的错误类型
```



with常常用在打开文件中,如下图所示,with    as  

```python
def test2():
    with open("1.txt", "w") as f:
        f.write("2222")
```





### 12、__ call __函数

```python
class Person:
    def __call__(self,name):
        print("__call__"+"hello"+name)

    def hello(self,name):
        print("hello"+name)


person=Person()
person("张三")#直接用实例化的对象加上括号就可以运行,本来是应该输入person.__call__("张三")的，相当于简略了很多步骤
person.hello("李四")#用实例化对象

>>>
__call__hello张三
hello李四

```





### 13、类中的super()函数

简单的理解，**super的作用就是执行父类的方法**

```python
class A:
    def p(self):
        print('A')
class B(A):
    def p(self):
        super().p()
B().p()

>>> A
```

更加复杂的理解，先看一下MRO序列

```python
class A:
    def p(self):
        print('A')
class B():
    def p(self):
        print('B')
class C(A,B):
    def p(self):
        print('C')
class D(C):
    def p(self):
        print('D')
a = A()
b = B()
c = C()
d = D()
#可以通过查看__mro__属性获得，例如A.__mro__
```

- A: (A, object)
- B: (B, object)
- C: (C, A, B, object)
- D: (D, C, A, B, object)

python中的class都直接或间接地继承Object class，第一个是自身，然后一步步往上找父类



super本身其实就是一个类，`super()`其实就是这个类的实例化对象，它需要接收两个参数 `super(class, obj)`,它返回的是`obj`的MRO中`class`类的父类

super()里面如果没有参数的话就说明找的是默认的上一级

```python
super(C, d).p()
#前面我们说过super的作用是 返回的是obj的MRO中class类的父类,在这里就表示返回的是d的MRO中C类的父类：
1.返回的是d的MRO：(D, C, A, B, object)
2.中C类的父类：A
```

那么`super(C, d)`就等价于`A`,那么`super(C, d).p()`会输出`A`

多继承

```python
class A:
    def __init__(self):
        print('A')
class B:
    def __init__(self):
        print('B')
class C(A,B):
    def __init__(self):
        super(C,self).__init__()
        print('C')
class D(B,A):
    def __init__(self):
        super(B,self).__init__()
        print('D')
print(C.__mro__)
print(D.__mro__)
print('initi C:')
c = C()
print('initi D:')
d = D()



>>>输出结果为
(<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>)
(<class '__main__.D'>, <class '__main__.B'>, <class '__main__.A'>, <class 'object'>)
initi C:
A
C
initi D:
A
D

```

### 14、repr 函数，包括类中的__ repr __

1、普通的repr函数

repr() 函数将对象转化为供解释器读取的形式。

- str()一般是将数值转成字符串，str()函数得到的字符串可读性好（故被print调用）
- repr()是将一个对象转成字符串显示，repr() 函数将对象转化为供解释器读取的形式。支持dict和list。repr是representation及描述的意思，不是对人的描述，而是对python机器的描述，也就是它会将某物返回一个它在python中的描述。对python友好。repr()函数得到的字符串通常可以用来重新获得该对象，通常情况下 obj==eval(repr(obj)) 这个等式是成立的。

```python
# 1、对待list上两者几乎没有任何区别
a = [1, 2, 3]
b = str(a)
print(str(b))  # [1, 2, 3]
print(type(b))  # <class 'str'>
print(eval(b))  # [1, 2, 3]
print(type(eval(b)))  # <class 'list'>


c = repr(a)
print(c)  # [1, 2, 3]
print(type(c))  # <class 'str'>
print(eval(c))  # [1, 2, 3]
print(type(eval(c)))  # <class 'list'>
```



2、类中的repr函数

```python
class My_func (object):
    def __init__(self,name):
        self.name=name

    def __str__(self):
        print("这是一个__str__函数")
        return self.name
    def __repr__(self):
        print("这是一个__repr__函数")
        return self.name
fun=My_func('doge')
print(fun)#这个内置方法会被调用当直接打印这个值的时候，并且__str__的优先级比__repr__高

>>>
这是一个__str__函数
doge

```

```python
class My_func (object):
    def __init__(self,name):
        self.name=name
    def __repr__(self):
        print("这是一个__repr__函数")
        return self.name

b=My_func("wangming")
print(b)

>>>
这是一个__repr__函数
wangming
```





### 15、类型注解

Python 中的类型注解——变量名后面加冒号标明变量类型，用法：

```python
var: type = value
    
'''
这是 Python 3.5 中引入的 Type Annotation，是一种注解，用来提示变量的类型。其中
var 为要定义的变量；
type 为该变量期待的类型；
value 为赋给该变量的值。
'''


#这种用法本质上和 var = value 相同，只是加上了 var 的类型说明。例如：
a: int = 10
a=10
#上面两式的意义是相同的

#也用在函数参数中
def func(arg: int)


需要注意的是，类型注解只是一种提示，并非强制的，Python 解释器不会去校验 value 的类型是否真的是 type，它只是在提示调用者该参数的类型。例如：
a: str = 10  
这样是没有错的，python 解释器在执行时会把 a 当作 int 来操作，a被指定str数据类型但是又用int类型给赋值
```



### 16、enumerate()函数

- 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值，既一般情况下对一个列表或数组既要遍历**索引**又要**遍历元素**；
- 多用于在for循环中得到计数；
- 返回的是一个enumerate对象

用法如下：

**enumerate(sequence, [start=0])**

- sequence ： 一个序列、迭代器或其他支持迭代对象
- start – 下标起始位置



```python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
print(list(enumerate(seasons)))                # 默认下标从0开始
print(list(enumerate(seasons, start=1)))      # 下标从 1 开始


输出：
[(0, ‘Spring’), (1, ‘Summer’), (2, ‘Fall’), (3, ‘Winter’)]
[(1, ‘Spring’), (2, ‘Summer’), (3, ‘Fall’), (4, ‘Winter’)]
print(enumerate(seasons))#<enumerate object at 0x000001FBC6719F40>
print(type(enumerate(seasons)))#<class 'enumerate'>

#------------传统方法-----------------
i = 0
seq = ['one', 'two', 'three']
for element in seq:
	print(i, seq[i])
	i +=1
输出：
0 one
1 two
2 three

#-------------enumerate-----------方法
for i,j in enumerate('abc'):
    print( i,j)
输出：
0 a
1 b
2 c
print( type(i),type(j)) 
输出：
<class 'int'> <class 'str'> #保持原本的数据类型
```

### 17、collections的用法总结

**1、deque**用法总结

`deque`是一个双向列表，非常适用于队列和栈，因为普通的`list`是一个线性结构，使用索引访问元素时非常快，但是对于插入和删除就比较慢，所以`deque`可以提高插入和删除的效率，可以使用`list(a_deque)`将`deque`转换成`list`。

常用的方法：

- `append`：向列表尾部添加元素
- `appendLeft`：向列表头部添加元素
- `pop`：从列表尾部取出元素
- `popLeft`：从列表头部取出元素

```python
from collections import deque
a = deque([1, 2, 3])

print(a)#deque([1, 2, 3])
a.append(4)#deque([1, 2, 3, 4])
a.appendleft(0)#deque([0, 1, 2, 3, 4])
print(a.pop())#输出4(int类型)
print(a.popleft())#输出0(int类型)

print(list(a))#转变成普通类型的列表
```



deque 里面也可以保存其他的东西，和普通的列表一样

```python
#超过最大容量的时候会把最开始的元素给去掉
a=collections.deque(maxlen=3)
a.append((1,1))
a.append((2, 2))
a.append((3, 3))
a.append((4, 4))
print(a)#deque([(2, 2), (3, 3), (4, 4)], maxlen=3)

```



### 18、range()

```python
#range()是左开右闭
for i in reversed(range(0,4)):
    print(i)

    
#翻转一下
>>>
3
2
1
0
```

```python
for select in range(0, 120, 10):
    print(select)
# 
0
10
20
30
40
50
60
70
80
90
100
110


```



### 19、刷新输出

```python
for i in range(50):
    print("测试：\r",i,end="")
#这样输出就可以一直在同一个地方进行刷新输出
```





### 22、python的for...in...if...

Python中，for...[if]...语句一种简洁的构建List的方法，从for给定的List中选择出满足if条件的元素组成新的List，其中if是可以省略的。下面举几个简单的例子进行说明

```python
>>> ak=[0,1,2,3,4,5,6,7]
>>> new_t = [x for x in ak]
>>> new_t
[0, 1, 2, 3, 4, 5, 6, 7] # 这里是输出的new_t，将ak中的值赋值给new_t





>>> ak=[0,1,2,3,4,5,6,7]
>>> new_t = [x for x in ak]
>>> new_t
[0, 1, 2, 3, 4, 5, 6, 7]
>>> new_t2 = [x for x in ak if x%2==0]
>>> new_t2
[0, 2, 4, 6]
```



```python
b="wangming"
a=[b for i in range(7)]
print(a)
# ['wangming', 'wangming', 'wangming', 'wangming', 'wangming', 'wangming', 'wangming']
# 每进行一次for循环都执行一次b语句并将b语句的结果保存在里面
```



### 23、round函数的使用

使用方法：round(参数，保留的小数位)

该函数遵循四舍六入五成偶原则，被约的5前面的数值若为偶数时，则会舍弃5，若5前面的数值为奇数时则进一，若5后面有非0数值时则都会进一，round()中默认保留小数位是0，即当只传一位参数时默认返回的是整数

```python
print(round(3.545,2))
# 3.54

print(round(3.5455,2))
# 3.55

print（round(3.535,2)）
# 3.54

print(round(3.54))
# 4



```

### 24、python中::(双冒号的用法)和：(单冒号)

```python
a=np.arange(0,9)
print(a[::2])
#>>>[0 2 4 6 8]
#将a中的元素两个两个分组并取每组中第一个出来，也可以理解为每2个数中取一个。

print(a[::3])
#>>>[0 3 6]
#将a中的元素三个三个分组并取每组中第一个出来。

print(a[::-1])
#>>>[8 7 6 5 4 3 2 1 0]

print(a[::-2])
#>>>[8 6 4 2 0]
```

```python
import numpy as np
a=np.arange(1,51)
print(a[4::5])#[ 5 10 15 20 25 30 35 40 45 50]
# 从下标为4的元素开始每隔5个取得一个
```

下面是单冒号的用法

```python
import numpy as np
X = np.array([[ 0  1  2]
 			  [ 3  4  5]
 			  [ 6  7  8]
 			  [ 9 10 11]
 			  [12 13 14]
 			  [15 16 17]
 			  [18 19 20]])

print(X[:,1:3]) #取出第一个维度的所有，第二个维度的第一列和第二列(含左不含右)

'''
[[ 1  2]
 [ 4  5]
 [ 7  8]
 [10 11]
 [13 14]
 [16 17]
 [19 20]]
'''
```



### 25、zip函数

参数 iterable 为可迭代的对象，并且可以有多个参数。该函数返回一个以元组为元素的列表，其中第 i 个元组包含每个参数序列的第 i 个元素。返回的列表长度被截断为最短的参数序列的长度。只有一个序列参数时，它返回一个1元组的列表。没有参数时，它返回一个空的列表。

```python
import numpy as np
a=[1,2,3,4,5]
b=(1,2,3,4,5)
c=np.arange(5)
d="zhang"
zz=zip(a,b,c,d)
print(zz)
print(tuple(zz))

输出：
<zip object at 0x000002383DCA0040>
((1, 1, 0, 'z'), (2, 2, 1, 'h'), (3, 3, 2, 'a'), (4, 4, 3, 'n'), (5, 5, 4, 'g'))
# 取每一个列表的第一个元素并且将其变成元祖

# 一般情况下，我们会通过遍历的方式取出这些东西
for a,b,c,d in zz:
    print("a=",a,"b=",b,"c=",c,"d=",d)     #输出各自的值
    print(type(a),type(b),type(c),type(d)) #输出各自的类型
    
'''
a= 1 b= 1 c= 0 d= z
<class 'int'> <class 'int'> <class 'numpy.int32'> <class 'str'>
a= 2 b= 2 c= 1 d= h
<class 'int'> <class 'int'> <class 'numpy.int32'> <class 'str'>
a= 3 b= 3 c= 2 d= a
<class 'int'> <class 'int'> <class 'numpy.int32'> <class 'str'>
a= 4 b= 4 c= 3 d= n
<class 'int'> <class 'int'> <class 'numpy.int32'> <class 'str'>
a= 5 b= 5 c= 4 d= g
<class 'int'> <class 'int'> <class 'numpy.int32'> <class 'str'>
''' 



# 第二种方式取出东西
------------------------
for i in zz:
    print(i)
    
(1, 1, 0, 'z')
(2, 2, 1, 'h')
(3, 3, 2, 'a')
(4, 4, 3, 'n')
(5, 5, 4, 'g')

```

​        zip(a,b)方法的工作原理是创建出一个迭代器，该迭代器可产生出元组（x,y）,这里的x取自数组a,而y取自序列b，当其中某个输入数组中没有元素可以继续访问迭代时，整个迭代过程结束。因此，整个迭代的长度取决于最短数组长度

```python
l1 = [1,2,3,4,5]
l2 = ['a','b','c','d','e','f']
for x,y in zip(l1,l2):
    print(x,y)
   
  结果：
1 a
2 b
3 c
4 d
5 e
```

也可以以最长的来

```python
from itertools import zip_longest
# 注意在这个地方导入的就和上面是不一样的了

l1 = [1,2,3,4,5]
l2 = ['a','b','c','d','e','f']
for i in zip_longest(l1,l2):
    print(i)
  
 结果：
(1, 'a')
(2, 'b')
(3, 'c')
(4, 'd')
(5, 'e')
(None, 'f')

```

### 26、忽略警告信息(warnings模块)

```python
import warnings
warnings.filterwarnings('ignore')
```

### 27、iter()函数和next()函数

**iter()** :我们使用iter()函数可以获取可迭代对象身上的迭代器，即将容器类型或者序列类型转为迭代器对象，生成迭代器。

**next()**: 返回迭代器的下一个项目

```python
#当我们已经迭代完最后⼀个数据之后，再次调⽤next()函数会抛出 StopIteration的异常 ，来告诉我们所有数据都已迭代完成，不⽤再执⾏ next()函数了。


it = iter([1,2,3,4,5]) # 输入的是一个列表
# 循环
while True:
    try:
        # 获得下一个值
        x = next(it)
        print(x)
    except StopIteration:
        # 遇到StopIteration就退出循环
        break
# 输出：
# 1
# 2
# 3
# 4
# 5
```

### 28、from package import *

```python
# 比如用这个就相当于把这个库里面所有的函数都变到现在这个库里面
from sympy import *

```



### 29、eval函数

去掉参数最外侧引号并且执行余下语句的函数。

eval()函数用于执行一个字符串表达式，并且返回该表达式的值。也就是说**就是说：将字符串当成有效的表达式 来求值并返回计算结果。**

**eval函数就是实现list、dict、tuple与str之间的转化，同样str函数把list，dict，tuple转为为字符串**

```python
eval(expression[, globals[, locals]])

expression – 表达式。
globals – 变量作用域，全局命名空间，如果被提供，则必须是一个字典对象。
locals  – 变量作用域，局部命名空间，如果被提供，可以是任何映射对象。
返回值：返回表达式计算结果
```

先了解**命名空间**的作用

**定义**：名称到对象的映射。python是用命名空间来记录变量的轨迹的，命名空间是一个dictionary，键是变量名，值是变量值。各个命名空间是独立没有关系的，一个命名空间中不能有重名，但是不同的命名空间可以重名而没有任何影响。

**分类**：python程序执行期间会有2个或3个活动的命名空间（函数调用时有3个，函数调用结束后2个）。按照变量定义的位置，可以分为以下3类。

	- **local**:局部命名空间，这个是每个函数拥有的命名空间，记录了函数中定义的所有变量，包括函数的入参以及内部定义的局部变量
	- **global**:全局命名空间，每个模块加载执行时创建的，记录了模块中定义的变量，包括模块中定义的函数、类、其他导入的模块
	- **Built-in**:python自带的内建命名空间，任何模块都可访问

**生命周期**：

- Local（局部命名空间）在函数被调用时才被创建，但函数返回结果或抛出异常时被删除。（每一个递归函数都拥有自己的命名空间）。
- Global（全局命名空间）在模块被加载时创建，通常一直保留直到python解释器退出。
- Built-in（内建命名空间）在python解释器启动时创建，一直保留直到解释器退出。

python解释器加载阶段会创建出**内建命名空间、模块的全局命名空间，局部命名空间**是在运行阶段函数被调用时动态创建出来的，函数调用结束动态的销毁的。

python的全局命名空间存储在一个叫**globals()**的dict对象中；局部命名空间存储在一个叫**locals()**的dict对象中。可以用print (locals())来查看该函数体内的所有变量名和变量值。



当eval()参数中的两个都不为空时：先查找locals参数，再查找globals参数。

**使用演示**

可以理解为将双引号去掉，使用双引号里面的语法

```python
#1.eval无参实现字符串转化
s = '1+2+3*5-2'
print(eval(s))  #16
 
#2.字符串中有变量也可以
x = 1
print(eval('x+2'))  #3
 
#3.字符串转字典
b = eval("{'name':'linux','age':18}")
print(eval(b))  # 输出结果：{'name':'linux','age':18}
print(type(b))  # <class 'dict'>

 
#4.eval传递全局变量参数,注意字典里的:age中的age没有带引号，说明它是个变量，而不是字符串。
#这里两个参数都是全局的
print(eval("{'name':'linux','age':age}",{"age":1822}))
#输出结果：{'name': 'linux', 'age': 1822}
print(eval("{'name':'linux','age':age}",{"age":1822},{"age":1823}))
#输出结果：{'name': 'linux', 'age': 1823}
 
#eval传递本地变量，既有global和local时，变量值先从local中查找。
age=18
print(eval("{'name':'linux','age':age}",{"age":1822},locals()))
#输出结果：{'name': 'linux', 'age': 18}
print("-----------------")
 
print(eval("{'name':'linux','age':age}"))
```



**应用1：**作用于input函数中，将字符串变为字符型。我们在从键盘输入数据时，Python接收的是字符串类型，这时我们可以使用eval()函数，将输入的数据进行还原。

```python
a = input("请输入一个数字:")
print(type(a))
b = eval(a)
print(type(b))
# 输出如下:
# 请输入一个数字:8
# <class 'str'>
# <class 'int'>
```



eval的使用与风险：

```python
eval虽然方便，但是要注意安全性，可以将字符串转成表达式并执行，就可以利用执行系统命令，删除文件等操作。
```



### 30、lambda()函数



```python

to_pi = lambda a: a if a < 180 else (a-360)

def to_pi(a):
    return a if a < 180 else (a-360)
```

### 31、map()函数

map函数的原型是**map(function, iterable, …)**

**参数function**传的是一个函数名，可以是python内置的，也可以是自定义的。
**参数iterable**传的是一个可以迭代的对象，例如列表，元组，字符串这样的

这个函数的意思就是将function应用于iterable的每一个元素，结果以列表的形式返回

```python
# 1、可迭代对象为1个的情况下
def mul(x):
    return x*x

n=[1,2,3,4,5]
res=map(mul,n)
print(res)
# 输出:
# [1, 4, 9, 16, 25]


# 2、可迭代对象为多个情况下
def add(x,y,z):
    return x,y,z

list1 = [1,2,3]
list2 = [1,2,3,4]
list3 = [1,2,3,4,5]
res = map(add, list1, list2, list3)# add函数输入的x,y,z分别是分别是list1/list2/list3中的第一个数字
print(res)

# 输出：
# [(1, 1, 1), (2, 2, 2), (3, 3, 3), (None, 4, 4), (None, None, 5)]
```

### 32、jitclass装饰器

可以快速节省大量的计算时间

Numba通过numba.jitclass（）装饰器支持类的代码生成。 可以使用此装饰器标记类以进行优化，同时指定每个字段的类型。 我们将结果类对象称为jitclass。 jitclass的所有方法都被编译成nopython函数。 jitclass实例的数据在堆上作为C兼容结构分配，以便任何已编译的函数都可以绕过解释器直接访问底层数据。

```python
import numpy as np
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types

spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class Bag(object):
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] = val
        return self.array
```

### 33、raise 异常处理

一、基本原理

python的异常机制主要依赖**try 、except、else、finally **和**raise**五个关键字

**try**关键字后面放置可能引发异常的代码

**except**之后对应的是异常类型和一个代码块，用于表明该except块处理这种类型的代码块

在多个except块之后可以放一个**else**块，表明程序不出现异常时还要继续执行else块

最后还可以跟一个**finally**块，**finally**块用于回收在try块里打开的物理资源，异常机制会保证finally块总被执行

**raise**用于引发一个实际的异常，raise可以单独作为一个语句使用，引发一个具体的异常对象。

python中内置异常类的层次结构如下：

```python
BaseException  # 所有异常的基类
 +-- SystemExit  # 解释器请求退出
 +-- KeyboardInterrupt  # 用户中断执行(通常是输入^C)
 +-- GeneratorExit  # 生成器(generator)发生异常来通知退出
 +-- Exception  # 常规异常的基类
      +-- StopIteration  # 迭代器没有更多的值
      +-- StopAsyncIteration  # 必须通过异步迭代器对象的__anext__()方法引发以停止迭代
      +-- ArithmeticError  # 各种算术错误引发的内置异常的基类
      |    +-- FloatingPointError  # 浮点计算错误
      |    +-- OverflowError  # 数值运算结果太大无法表示
      |    +-- ZeroDivisionError  # 除(或取模)零 (所有数据类型)
      +-- AssertionError  # 当assert语句失败时引发
      +-- AttributeError  # 属性引用或赋值失败
      +-- BufferError  # 无法执行与缓冲区相关的操作时引发
      +-- EOFError  # 当input()函数在没有读取任何数据的情况下达到文件结束条件(EOF)时引发
      +-- ImportError  # 导入模块/对象失败
      |    +-- ModuleNotFoundError  # 无法找到模块或在在sys.modules中找到None
      +-- LookupError  # 映射或序列上使用的键或索引无效时引发的异常的基类
      |    +-- IndexError  # 序列中没有此索引(index)
      |    +-- KeyError  # 映射中没有这个键
      +-- MemoryError  # 内存溢出错误(对于Python 解释器不是致命的)
      +-- NameError  # 未声明/初始化对象 (没有属性)
      |    +-- UnboundLocalError  # 访问未初始化的本地变量
      +-- OSError  # 操作系统错误，EnvironmentError，IOError，WindowsError，socket.error，select.error和mmap.error已合并到OSError中，构造函数可能返回子类
      |    +-- BlockingIOError  # 操作将阻塞对象(e.g. socket)设置为非阻塞操作
      |    +-- ChildProcessError  # 在子进程上的操作失败
      |    +-- ConnectionError  # 与连接相关的异常的基类
      |    |    +-- BrokenPipeError  # 另一端关闭时尝试写入管道或试图在已关闭写入的套接字上写入
      |    |    +-- ConnectionAbortedError  # 连接尝试被对等方中止
      |    |    +-- ConnectionRefusedError  # 连接尝试被对等方拒绝
      |    |    +-- ConnectionResetError    # 连接由对等方重置
      |    +-- FileExistsError  # 创建已存在的文件或目录
      |    +-- FileNotFoundError  # 请求不存在的文件或目录
      |    +-- InterruptedError  # 系统调用被输入信号中断
      |    +-- IsADirectoryError  # 在目录上请求文件操作(例如 os.remove())
      |    +-- NotADirectoryError  # 在不是目录的事物上请求目录操作(例如 os.listdir())
      |    +-- PermissionError  # 尝试在没有足够访问权限的情况下运行操作
      |    +-- ProcessLookupError  # 给定进程不存在
      |    +-- TimeoutError  # 系统函数在系统级别超时
      +-- ReferenceError  # weakref.proxy()函数创建的弱引用试图访问已经垃圾回收了的对象
      +-- RuntimeError  # 在检测到不属于任何其他类别的错误时触发
      |    +-- NotImplementedError  # 在用户定义的基类中，抽象方法要求派生类重写该方法或者正在开发的类指示仍然需要添加实际实现
      |    +-- RecursionError  # 解释器检测到超出最大递归深度
      +-- SyntaxError  # Python 语法错误
      |    +-- IndentationError  # 缩进错误
      |         +-- TabError  # Tab和空格混用
      +-- SystemError  # 解释器发现内部错误
      +-- TypeError  # 操作或函数应用于不适当类型的对象
      +-- ValueError  # 操作或函数接收到具有正确类型但值不合适的参数
      |    +-- UnicodeError  # 发生与Unicode相关的编码或解码错误
      |         +-- UnicodeDecodeError  # Unicode解码错误
      |         +-- UnicodeEncodeError  # Unicode编码错误
      |         +-- UnicodeTranslateError  # Unicode转码错误
      +-- Warning  # 警告的基类
           +-- DeprecationWarning  # 有关已弃用功能的警告的基类
           +-- PendingDeprecationWarning  # 有关不推荐使用功能的警告的基类
           +-- RuntimeWarning  # 有关可疑的运行时行为的警告的基类
           +-- SyntaxWarning  # 关于可疑语法警告的基类
           +-- UserWarning  # 用户代码生成警告的基类
           +-- FutureWarning  # 有关已弃用功能的警告的基类
           +-- ImportWarning  # 关于模块导入时可能出错的警告的基类
           +-- UnicodeWarning  # 与Unicode相关的警告的基类
           +-- BytesWarning  # 与bytes和bytearray相关的警告的基类
           +-- ResourceWarning  # 与资源使用相关的警告的基类。被默认警告过滤器忽略。
```

异常处理流程：

首先，执行 try 子句（在关键字 try 和关键字 except 之间的语句）。如果没有异常发生，忽略 except 子句，try 子句执行后结束。
如果在执行 try 子句的过程中发生了异常，那么 try 子句余下的部分将被忽略。如果异常的类型和 except 之后的名称相符，那么对应的 except 子句将被执行。如果一个异常没有与任何的 except 匹配，那么这个异常将会传递给上层的 try 中。

```python
try:
    # {执行代码}
except:
    # {发生异常时执行的代码}
else:
    # {没有异常时执行的代码}
finally:
    # {不管有没有异常都会执行的代码}
```



1、异常类的基本使用方法

```python
import sys

try:
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    c = a / b
    print("您输入的两个数相除的结果是：", c)
except IndexError:
    print("索引错误，运行程序时输入的参数个数不够")
except ValueError:
    print("数值错误：程序只能接收整数参数")
except ArithmeticError:
    print("算术错误")
except Exception:
    print("未知错误")
```

2、多异常捕获

```python
import sys

try:
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    c = a / b
    print("您输入的两个数相除的结果是：", c)
except (IndexError, ValueError, ArithmeticError):
    print("程序发生了数组越界、数字格式异常、算术异常之一")
except Exception:
    print("未知错误")
```

3、访问异常信息

```python
def foo():
    try:
        fis = open("a.txt");
    except Exception as e:
        # 访问异常的错误编号和详细信息
        print(e.args)  # (2, 'No such file or directory')
        # 访问异常的错误编号
        print(e.errno)  # 2
        # 访问异常的详情信息
        print(e.strerror)  # No such file or directory

foo()
```

4、运用else

- 无异常情况下

```python
try:
    fh = open("testfile.txt", "w")
    fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print("Error: 没有找到文件或读取文件失败")
else:
    print("内容写入文件成功")
    fh.close()
# 输出：内容写入文件成功
```

- 无异常情况下

```python
try:
    fh = open("testfile.txt", "r")
    fh.write("这是一个测试文件，用于测试异常!!")
except IOError:
    print("Error: 没有找到文件或读取文件失败")
else:
    print("内容写入文件成功")
    fh.close()
# 输出：Error: 没有找到文件或读取文件失败

```

5、使用finally回收资源

有时候在try块里打开一些物理资源，如数据库连接、网络连接和磁盘文件等，这些资源都必须被显示回收说到资源回收可能会联想到Python的垃圾回收机制，其实还是不一样的，Python的垃圾回收机制不会回收任何物理资源，只能回收堆内存中对象所占用的内存为了保证一定能回收在try块里打开的物理资源，异常处理机制提供了finally块，不管try块中的代码是否出现异常，也不管哪一个except块被执行，甚至在try块或者except块中被执行了return语句，finally块也总会被执行

```python
# 注意不要在finally块中使用return或者raise等导致终止的语句，因为无论如何，finally中的语句是要强制执行的，即使已经出现了return True，但是程序是不会停止的，知道finally中的语句执行。一旦finally中使用了return 或者 raise语句，会导致try、except块中的return、raise语句失效。
def test():
    try:
        # 因为finally块中包含了return语句
        # 所以下面的return语句失去作用
        return True
    finally:
        return False
a = test()
print(a)  # False


# 如果程序改成下面这样
def test():
    try:
        # 因为finally块中包含了return语句
        # 所以下面的return语句失去作用
        return True
    finally:
        print("nihao")
a = test()
print(a)
# 输出：
# nihao
# True
```



6、raise的使用方法

raise可以手动引发异常

raise语句的基本语法格式为:

```python
raise [exceptionName [(reason)]] # []不是程序中运行的
# []中的都是可选参数
# exceptionName是抛出异常的名称(异常名称必须是前面定义的那些)，后面reason是这个异常的相关描述

# 例如：
raise Exception
# 上面是执行了引发异常的基类
```

下面是一些使用方法

```python
# case 1
raise
#   File "E:/wm/桌面/sumo/traci_study/test.py", line 108, in <module>
#     raise
# RuntimeError: No active exception to reraise


# case 2
raise ZeroDivisionError
print(a)
#   File "E:/wm/桌面/sumo/traci_study/test.py", line 108, in <module>
#     raise ZeroDivisionError
# ZeroDivisionError


# case 3
raise ZeroDivisionError("除数不能为零")
print(a)
#   File "E:/wm/桌面/sumo/traci_study/test.py", line 108, in <module>
#     raise ZeroDivisionError("除数不能为零")
# ZeroDivisionError: 除数不能为零
```

当然，我们手动让程序引发异常，很多时候并不是为了让其崩溃。事实上，raise 语句引发的异常通常用 try except（else finally）异常处理结构来捕获并进行处理。例如：

```python
# 手动引发异常并且使用except来捕获异常
try:
    a = input("输入一个数：")
    # 判断用户输入的是否为数字
    if (not a.isdigit()):
        raise ValueError("a 必须是数字")
except ValueError as e:
    print("引发异常：", repr(e))

# 输入一个数：b
# 引发异常： ValueError('a 必须是数字')


# 当在没有引发过异常的程序使用无参的 raise 语句时，它默认引发的是 RuntimeError 异常。例如：
try:
    a = input("输入一个数：")
    if not a.isdigit():
        raise
except RuntimeError as e:
    print("引发异常：",repr(e))
# 输入一个数：a
# 引发异常： RuntimeError('No active exception to reraise')
```



二、应用

1、NotImplementedError

```python
class FatherClass:
  def func(self):
    raise NotImplementedError("ERROR: func not implemented!")
 
class ChildClass(FatherClass):
  def func(self):# 把子类的这个函数去除掉就会报错
    print("hello world!")
 
obj = ChildClass()
obj.func()
```





### 34、callable()函数

**Python3**中的内置函数callable接受一个对象参数，如果此对象参数看起来可调用，则callable函数返回True，否则返回False**。**如果返回True，则调用仍有可能失败；但如果返回False，则调用对象将永远不会成功。

```python
# 对于
print(callable(0))  # False
print(callable("wangming"))  # False

# 对于函数
def add(a,b):
    return a+b  # 即使函数不返回值下面依然是True
print(callable(add)) # True 函数返回True

# 对于类
## case1:
class A:
    def method(self):
        return 0
print(callable(A)) # True 类返回True

# 对于类的实例
a=A()
print(callable(a)) # False 类的实例中没有实现_call_,返回False

## case2:
class B:
    def __call__(self):
        return 0
print(callable(B)) # True

b=B()
print(callable(b)) # True
```



### 35、python中的not常见用法

python中的not主要有以下几个用法

- 一、用于判断变量是否为None
- 二、判断有类型的变量是否为空
- 三、和关键字in搭配，作包含关系的判断

```python
# 用法1：
a = None
if not a:
	print('a为空')
    
# 用法2
# 数字
零 = 0
if not 零:
    print(type(零),'0')

# 列表
lst = []
if not lst:
    print(type(lst),'列表为空')   # <class 'list'> 列表为空
# 布尔值
bool =  False
if not bool:
    print(type(bool),'布尔值为false') # <class 'bool'> 布尔值为false
# 字典
dict = {}
if not dict:
    print(type(dict),'字典为空')  # <class 'dict'> 字典为空
# 字符串
str = ''
if not str:
    print(type(str),'字符串为空') # <class 'str'> 字符串为空
# 集合
set = set()
if not set:
    print(type(set),'集合为空')  # <class 'set'> 集合为空
# 元组
tuple = tuple()
if not tuple:
    print(type(tuple),'元组为空') # <class 'tuple'> 元组为空

    
# 用法3：和关键字in搭配，做包含关系的判断
content = '''家珍一直扑到天黑，我怕夜露伤着她，硬把她背到身后。家珍让我再背她到村口去看看，到了村口，我的衣领都湿透了，家珍哭着说:有庆不会在这条路上跑来了。我看着那条弯曲着通向城里的小路，听不到我儿子赤脚跑来的声音，月光照在路上，像是撤满了盐'''

keys = ['儿子', '家珍', '到了村口', '一个', '城里小路']

# 推导式写if else要写在循环前面
res = ['否' if key not in content else '是' for key in keys]
print(res)
# 输出字符串组成的列表：['是', '是', '是', '否', '否']

dic = [{key: '否'} if key not in content else {key: '是'} for key in keys]
print(dic)
# 输出字典组成的列表： [{'儿子': '是'}, {'家珍': '是'}, {'到了村口': '是'}, {'一个': '否'}, {'城里小路': '否'}]
```



### 36、python中的None

一：None

　　None是python中的一个特殊的常量，表示一个空的对象。

　　数据为空并不代表是空对象，例如[],''等都不是None。

　　None有自己的数据类型NontType，你可以将None赋值给任意对象，但是不能创建一个NoneType对象。

 

二：False

　　python中数据为空的对象以及None对象在条件语句都作False看待：即 None，False，0，[]，""，{}，() 都相当于False。

　　

三：None的比较——用 is None 而不是 == None

　　因为None在Python里是个单例对象，一个变量如果是None，它一定和None对象指向同一个内存地址。

　　is运算判断两个对象在内存中的地址是否一致：



### 37、python中的assert

如果assert后面为真则继续往下进行，如果为假的话则立即停止程序

```python
s_age = input("请输入您的年龄:")
age = int(s_age)
assert 20 < age < 80
print("您输入的年龄在20和80之间")

# assert 后面为真没有影响
```

### 38、python中的del关键字

使用del语句，删除变量到对象的引用和变量名称本身，注意del语句作用在变量上而不是数据对象上

**可用于pycharm的调试控制台中用来删除在当前运行内存中的变量**

**python变量的本质**：我们在代码中写了a=1,其实是创建了int对象a，a指向int对象1，所以我们使用a的时候就是1

1、变量

```python
a=1       # 对象 1 被 变量a引用，对象1的引用计数器为1（计数器是由于python的GC完成的）
b=a       # 对象1 被变量b引用，对象1的引用计数器加1
c=a       #1对象1 被变量c引用，对象1的引用计数器加1
del a     #删除变量a，解除a对1的引用
del b     #删除变量b，解除b对1的引用
print(c)  #最终变量c仍然引用1
```

2、列表

```python
li = [1, 2, 3, 4, 5]  # 列表本身不包含数据1,2,3,4,5，而是包含变量：li[0] li[1] li[2] li[3] li[4]
first = li[0]     # 拷贝列表，也不会有数据对象的复制，而是创建新的变量引用
del li[0]
print(li)      # 输出[2, 3, 4, 5]
print(first)   # 输出 1
```

### 39、python中的is关键字

**关键字is在python中时用来验证对象之间的同一性，它的要求比“==”操作更加严格**

python中的对象有三个属性：**身份（id） 类型（type）值（value）**

身份可以用id()函数查询，类型可以用type（）函数查询、值的话用print打印出来就可以看到。

只有这三个属性都完全一致时，用is判断时才会返回真值。而“==”操作只需要value值相同就可以成立。



**python中的id函数用于获取对象的内存地址**

### 40、python中的filter函数

filter() 函数用于 过滤 可迭代对象中不符合条件的元素，返回由符合条件的元素组成的新的迭代器。filter() 函数把传入的函数依次作用于每个元素，然后根据返回值是 True 还是 False，来决定保留或丢弃该元素。

```python
filter(function, iterable)
```

参数说明：

**(1) function**：用于实现判断的函数，可以为 None。
**(2) iterable**：可迭代对象，如列表、range 对象等。
**(3) 返回值:**返回一个迭代器对象。



通常情况下我们需要将输出转换成列表类型并且返回



```python
# 1、应用1
list1 = [('小明', 600), ('小刚', 602), ('小王', 800), ('小李', 400)]
print(list(filter(lambda x: 600 <= x[1] <= 700, list1)))
# [('小明', 600), ('小刚', 602)],输入所有分数在600分与700分之间的学生

# 2、去除序列中所有值为假的元素
# 如果将 filter() 函数的第一个参数 function 的值设置为 None，就会默认去除序列中所有值为假的元素，如 None、False、0、’’、()、[] 和 {}等，代码如下：
list1 = ['', False, 1, 0, None, [], 3, 4, [1, 23]]
print(list(filter(None, list1)))  # [1, 3, 4, [1, 23]]
```

### 41、sorted函数

```python
# 接受一个可迭代对象，常常与lambda函数一起使用
students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]  
sorted(students, key=lambda student : student[2])   # sort by age  
 
# 输出如下：
[('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
```

### 42、python中的 __ doc __ 方法

__doc__方法是python的内置方法之一，该方法通常会输出指定对象中的注释部分。

```python
class Debug:
    """
    This is a class for debugging
    """
    def __init__(self):
    	"""
    	This funtion only has one property
		"""
        self.x = 5
        
        
# debug  注意后面输入输出的内容
main = Debug()
print(main.__doc__)  			# This is a class for debugging  
print(main.__init__.__doc__) 	# This funtion only has one property
```



```python
# 下面相当于直接在python文件中输出doc
# 这样会直接输出整个文件最前面的注释内容
示例：
print(__doc__)
# Example script to generate traffic in the simulation
```

### 43、Python 字符串前r、b、u和f的前缀作用及用法

1、字符串前面加上u

```python
作用：表示后面字符串以Unicode进行编码，一般用在中文字符串前面防止出现乱码
a=u"王明"
```

2、字符串前面加上r，表示为普通字符串，转义字符也被当成普通的字符串来处理

```python
a=r"\n\n\n"
```

3、字符串前面加上b

b" "前缀表示：后面字符串是bytes 类型。

```python
response = b'Hello World!' b' ' 表示这是一个 bytes 对象
```

4、字符串前面加上f

以 f开头表示在字符串内支持大括号内的python 表达式

```python
import time

t0 = time.time()
time.sleep(1)
name = "processing"
print(f'{name} done in {time.time() - t0:.2f} s')
```

### 44、python中的next函数

函数必须接收一个可迭代对象参数，每次调用的时候，返回可迭代对象的下一个元素。如果所有元素均已经返回过，则抛出StopIteration 异常。

```python
a = [1,2,3,4]
it = iter(a)
print (next(it))
print (next(it))
print (next(it))
print (next(it))
print (next(it))  # 报错StopIteration

a = [1,2,3,4]
it = iter(a)
print (next(it,5))
print (next(it,5))
print (next(it,5))
print (next(it,5))
print (next(it,5))
print (next(it,5)) #所有元素都已经返回，现在就返回default指定的默认值5了
```





## 二、内置的库

### 1、pprint()

美化控制台输出的库

import pprint
pp = pprint.PrettyPrinter(width = 80)

函数里面可传参数如下：

- `width` 定义最大的宽度（如果超过这个宽度会自动换行）（默认的宽度为80）
- `indent` 定义缩减的空格数（默认为1）
- `compact` 如果为 `True` 则将同一种元素压缩到一行（默认为 `False` ）
- `depth` 定义最大的嵌套数。如果列表中超过一定的深度就以 `...` 显示

```python
# 1、width
import pprint
ls = ['abc', 'def', 'ghj', ['abc', 'def']]
pp = pprint.PrettyPrinter(width = 30)
pp.pprint(ls)
print(ls)
>>>
['abc',
 'def',
 'ghj',
 ['abc', 'def']]#使用pprint的输出
['abc', 'def', 'ghj', ['abc', 'def']]#使用普通的输出


#2、indent
pp = pprint.PrettyPrinter(width = 30,indent=4)
pp.pprint(ls)
[   'abc',
    'def',
    'ghj',
    ['abc', 'def']]#输出的时候每一个元素前面缩进了4个空格

# 3、depth
import pprint
ls = ['abc', 'def', 'ghj', ['abc', ['abc', ['abc', 'def'], 'def'], 'def']]
pp= pprint.PrettyPrinter(indent = 2, depth = 2, width=20)
pp.pprint(ls)

>>>#ls是一个嵌套的列表，depth表示最多显示嵌套列表的两层
[ 'abc',
  'def',
  'ghj',
  [ 'abc',
    [...],
    'def']]
```



### 2、sys库

1、基本用法

```python
import sys

# 1、在命令行中输入参数
a=sys.argv# 在命令行中输入python test.py zsl wm 9来执行
print(a)# ['test.py', 'zsl', 'wm', '9'],第一个参数是文件的路径，后面是我们的输入
print(type(a))#<class 'list'>

# 2、正常退出程序
sys.exit("Arriving at the destination, the program exits normally")

# 3、输出操作系统能够承载的最大int值
print(sys.maxsize)  # 9223372036854775807
```

2、sys.path.append()

当我们导入一个模块时：import  xxx，默认情况下python解析器会搜索当前目录、已安装的内置模块和第三方模块，搜索路径存放在**sys模块的path**中：

我们利用通过下面命令来找到当前python的搜索目录

```python
import sys
print(sys.path) # 这是一个列表，列表里面保存的是当前python的搜索命令
```

然而当我们当前程序的路径与导入模块的路径不在同一个文件夹的时候

```python
import sys
sys.path.append(’引用模块的地址')
```



### 3、math库

#### 1、math.inf

```python
import math
print(math.inf) # 返回浮点数正无穷大
print(type(math.inf)) # <class 'floa
```



Uncertainty_estimation_for_Cross-dataset_performance_in_Trajectory_prediction

Multimodal_Trajectory_Prediction_Conditioned_on_Lane_Graph_Traversals

ReCoG:A_Deep_Learning_Framework_with_Heterogeneous_Graph_for_Interaction_Aware_Trajectory_Prediction

### 4、optparse库

这是一个命令行参数解析器，和下面5差不多，在python中已经弃用了

```python
from optparse import OptionParser

parser = OptionParser()

# 通过add_option来添加命令行参数，命令行参数由参数名和参数属性组成
# 可输入两个：短参数名+长参数名  短参数名和长参数名分别用-和--来表示
parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose",
                  help="don't print status messages to stdout")

# 1、简单用法
# 通过下面来解析命令行参数,剖析并返回一个字典和列表
(options, args) = parser.parse_args()
# options是一个字典，字典中的key就是所有输入命令行参数的dest值,字典的value就是用户输入的值或者是对应的default值
# 其中，dest就相当于给他起了一个别名，如果没有的话就默认是长参数值file
# args,它是一个由 positional arguments(位置参数) 组成的列表。位置参数就是传入的其他参数值


print(options)  # {'filename': None, 'verbose': True}
print(args)     # []

# 2、传入相关的参数
fakeArgs = ['-f','file.txt','-q','how are you', 'arg1', 'arg2']
# 我们传入了相关的参数
# 也就相当于我们在命令行中输入了以下命令(经过验证，下述完全正确)
# python test.py -f file.txt -q 'how are you' arg1 arg2
# 也就是相当于我们传入了假参数，让其来解析我们自己的假参数

op , ar = parser.parse_args(fakeArgs)
print(op)  # {'filename': 'file.txt', 'verbose': False}
print(ar)  # ['how are you', 'arg1', 'arg2']


# 3、add_option()函数的深入分析
# add_option()
# 参数说明：
# action: 存储方式，分为三种store、store_false、store_true
# type: 类型
# dest: 存储的变量
# default: 默认值
# help: 帮助信息

# action有很多参数，不指定的话默认就是store
# store:这个是默认，表示如果相应的参数在命令行中如果被传入，那么在解析的option这个字典中，key=dest value=传入值，没有传入就是保存None
# store_false：如果命令行中传入该参数，则在option的value就保存False，没有传入就保存None (这个地方要分清楚),如果想要没有传入保存成True, 就在parser.add_option中增加一个默认选项default=True，这样就达到非黑即白的效果

# store_true 就与前面恰好相反

```





### 5、argparse库

我们常常可以把argparse的使用简化成下面四个步骤

1：import argparse

2：parser = argparse.ArgumentParser()

3：parser.add_argument()

4：parser.parse_args()

**上面四个步骤解释如下：首先导入该模块；然后创建一个解析对象；然后向该对象中添加你要关注的命令行参数和选项，每一个add_argument方法对应一个你要关注的参数或选项；最后调用parse_args()方法进行解析；解析成功之后即可使用。**

```python
import argparse

# 通过下面的设置就可以从命令行中出入
parser = argparse.ArgumentParser(description='命令行请输入数字')  # 创建一个类的对象,里面是提示信息
parser.add_argument('--n_episodes', type=int, default=500)  # 可选参数
parser.add_argument('--gamma', type=float, default=0.95) #type可选的有list, str, tuple, set, dict
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()
#把命令行的参数打包成类似于字典的数据类型
print(args)         # Namespace(gamma=0.95, lr=0.001, n_episodes=500, seed=0)
print(type(args))   #<class 'argparse.Namespace'>
#通过下面的命令提取参数，注意不要加上前缀"--"
print(args.gamma)   #0.95 

#'--seed'前面加上的'--'表示他是可选参数，也就是说即使运行时没有传入这个参数，他也不会报错
#'-'加上一个横杠的代表关键字参数
```

上面运行可以输入：**python test.py --n_episodes=1 --gamma=2 --seed=3 --lr=4**

上面虽然繁琐，但是增加了可读性，避免了位置的问题，默认是采用顺序的位置顺序，直接**python test.py 1 2 3 4**

那么1 2 3 4按照程序中的位置分别赋值给--n_episodes --gamma --seed --lr



如果在命令行中输入python test.py -h, 会输出以下信息

```python
usage: main.py [-h] [--n_episodes N_EPISODES] [--gamma GAMMA] [--seed SEED] [--lr LR]

命令行请输入数字

optional arguments:
  -h, --help            show this help message and exit
  --n_episodes N_EPISODES
  --gamma GAMMA
  --seed SEED
  --lr LR
```

一点扩展

```python
import argparse

# 1.创建解释器
parser = argparse.ArgumentParser(description="可写可不写，只是在命令行参数出现错误的时候，随着错误信息打印出来。")
# 2.添加需要的参数
parser.add_argument('-gf', '--girlfriend', choices=['jingjing', 'lihuan'])
# 参数解释
# -gf 代表短选项，在命令行输入-gf和--girlfriend的效果是一样的，作用是简化参数输入
#--girlfriend 代表完整的参数名称，可以尽量做到让人见名知意，需要注意的是如果想通过解析后的参数取出该值，必须使用带--的名称
# choices 代表输入参数的只能是这个choices里面的内容，其他内容则会保错
parser.add_argument('--house', type=int, default=0)
# 参数解释
# --house 代表参数名称
# type  代表输入的参数类型，从命令行输入的参数，默认是字符串类型
# default 代表如果该参数不输入，则会默认使用该值
parser.add_argument('food')
# 参数解释
# 该种方式则要求必须输入该参数
# 输入该参数不需要指定参数名称，指定反而报错，解释器会自动将输入的参数赋值给food

# 3.进行参数解析
args = parser.parse_args() 
print('------args---------',args)
print('-------gf-------', args.girlfriend)

```

parser.add_argument()里携带的主要参数 ，具体包括：

- **action** ：

- - store：存储参数的值，为默认选项。
  - store_const：存储被const命名参数指定的值，通常在选项中来指定一些标记。
  - store_true / store_false：布尔开关。可以2个参数对应一个变量。
  - parser.add_argument('--color',action='store_true') ,其中--color的值就可以是True和False
  - append：存储值到列表，该参数可以重复使用。
  - append_const：存储值到列表，存储值在参数的const部分指定。
  - version：输出版本信息然后退出。

- **default**：没有传递值时取默认值

- **dest：**在代码中解析后的参数名称

- **required:** 如果设置了required=True,则该参数为必填参数，不输入会报错

- **choices：**参数值只能从几个选项里面选择

- **nargs：**设置参数在使用可以提供的个数

- - nargs='n' 表示参数可设置具体的n个
  - nargs='*'  表示参数可设置零个或多个
  - nargs='+'             表示参数可设置一个或多个
  - nargs='?'              表示参数可设置零个或一个。

**type** : 参数类型，默认类型为字符串 ，还可以包括float,int类型

args分为可选参数（用`--`指定）和必选参数（不加`--`指定）



action='store_true'的用法试例代码

```python
import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--achoice', action='store_true')# 默认为False，有动作(运行时传参)的话就设置为True
parser.add_argument('--bchoice', action='store_false')
args = parser.parse_args()
print(args.achoice)
print(args.bchoice)
# 第一种情况
# 如果如果运行python test.py运行python test.py --achoice结果输出True
# 如果运行python test.py 结果输出False
# 也即是说运行时该变量有传参的话就将该变量设置为True
# 第二种情况则与第一种情况完全相反
```

### 6、time()库的使用方法

```python
 import time
 time_start = time.time()
 time_end = time.time()
 print('time cost', time_end - time_start, 's')
```

### 7、os库

```python
import os

# 1、将某一个文件夹下面所有的文件名变成一个列表
dir_path="dataset/train/ants"
#将这个文件夹(文件夹路径)下面所有的内容变成一个列表，列表里面保存的是每一个文件(该文件夹下面)的名字，以字符串的形式保存
img_path_list=os.listdir(dir_path)

# 2、判断某一个文件夹是否存在以及创建文件夹
import os
if not os.path.exists("./wm"):
     os.makedirs("wm/slmm")import os

# 1、将某一个文件夹下面所有的文件名变成一个列表
dir_path="dataset/train/ants"
#将这个文件夹(文件夹路径)下面所有的内容变成一个列表，列表里面保存的是每一个文件(该文件夹下面)的名字，以字符串的形式保存
img_path_list=os.listdir(dir_path)

# 2、判断某一个文件夹是否存在以及创建文件夹
import os
if not os.path.exists("./wm"):
     os.makedirs("wm/slmm")
```





1、按任意键暂停的操作

```python
import os
os.system("pause")   # 按任意键继续的操作
```

2、os.environ环境变量

```python
import os
a = os.environ  # 这是一个环境变量的字典
b = os.environ.keys()  # 输出环境变量的键
```

3、os.path.join()函数用法详解

```python
# 主要用于连接多个路径,会自动把两个路径中重复的部分给自动去除掉
import os.path as osp
a = osp.join('D:/wm/nihao/sl', 'D:/wm/nihao/sl/tuantuan')
print(a)
# D:/wm/nihao/sl/tuantuan

```

4、目录文件相关

```python
import os
# 判断当前路径下的是否是一个目录
print(os.path.isdir("./hello"))   # True
# 判断当前路径是否是一个文件
print(os.path.isfile("./hello"))  # False
# 创建一个名为"nihao"的目录


# 下面两个命令都可用于创建文件夹
os.makedirs("nihao")  # 创建了一个名为"nihao"的文件夹，递归创建文件夹，如果中间有一级不存在则就自动创建
os.mkdir("nihao") # 如果中间有一级不存在，并不会自动创建文件
 
```

5、os.system()

```python
# 1、通过os.system() 直接调用系统功能
os.system("calc")   # 调用操作系统的计算器
os.system("cmd")    # 调用操作系统的cmd
os.system('mstsc')  # 调用远程桌面连接

# 2、通过cmd调用系统功能
 # 1、调用操作系统cmd面板
 os.system("cmd")
 # 2、在cmd面板输入相关功能命令
 calc
 mstsc
 mspaint
```

### 8、warnings库

```python
import warnings
warnings.warn('这是自定义的警告消息', category=UserWarning) # 向命令行输出一个警告信息


import warnings
warnings.filterwarnings('ignore')  # 过滤掉警告消息
```

### 9、copy库

1、深拷贝与浅拷贝

```python
import copy
origin = [1, 2, [3, 4]]
cop1 = copy.copy(origin)
cop2 = copy.deepcopy(origin)  # 深拷贝获得了一个独立的个体
print(cop1)  # [1, 2, [3, 4]]
print(cop2)  # [1, 2, [3, 4]]

origin[2][0] = "hey!"
print(origin)  # [1, 2, ['hey!', 4]]
print(cop1)  # [1, 2, ['hey!', 4]]
print(cop2)  # [1, 2, [3, 4]]
```

### 10、pickle库

pickle.dump()

pickle模块可以将任意的对象序列化成二进制的字符串写入到文件中。还可以从文件中读取并且转为写入时候类型。

pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，

存储：pickle.dump(obj, file,[protocol=None])**

序列化对象，将对象obj保存到文件file中去。
参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。
file表示保存到的类文件对象，file必须有write()接口，file可以是一个以’w’打开的文件或者是一个StringIO对象，也可以是任何可以实现write()接口的对象。

```python
import pickle

#创建一个字典变量
data = {'a':[1,2,3],'b':('string','abc'),'c':'hello'}
print(data)

#以二进制方式来存储,rb,wb,wrb,ab
pic = open(r'E:\wm\桌面\GNN轨迹预测\testdata.pkl','wb')

#将字典数据存储为一个pkl文件
pickle.dump(data,pic)
pic.close()

#读取 pickle.load(file)
pic2 = open(r'E:\wm\桌面\GNN轨迹预测\testdata.pkl','rb')
data = pickle.load(pic2)
print(data)
print(type(data))·
```

### 11、random()库

1、random.choice(seq)函数

从非空序列中随机选择一个数据并返回，该序列可以是*list、tuple、str、set*。

```python
import random
print(random.choice('choice')) 

# choice 其中任意一个字母
```

2、random.shuffle(list)

将一个list里面数据给打乱，注意这样并不产生新的list，只是将原来的list打乱

```python
import random

x = [i for i in range(10)]
print(x)
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random.shuffle(x)
print(x)
[2, 5, 4, 8, 0, 3, 7, 9, 1, 6]
```

### 12、Itertools

迭代工具

1、Itertools.cycle()

```python
from itertools import cycle
a=["wangmign","shaolin","tuantuan"]
b = cycle(a)
for ele in b:
    print(ele)
# 结果会无限的输出结果的三个值


常常与next函数一块用,这样可以依次循环的输出相关的内容
for epoch in range(100):
    ele = next(b)
    print(ele)
```

### 13、csv库

```python
import csv
with open(log_path+"predictor_log.csv", 'w') as csv_file: # log_path是指定下来的路径
	writer = csv.writer(csv_file) 
	writer.writerow(['steps', 'loss', 'lr', 'ADE', 'FDE'])  # 往csv文件中加入一行
```

### 14、collections

1、defaultdict

默认字典，python中自带的字典如果key值不存在的话，会出现报错，而运用default的话，key值不存在的话就返回一个默认值。

dict =defaultdict( factory_function)

这个factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0，如下举例

```python
from collections import defaultdict
 
dict1 = defaultdict(int)
dict2 = defaultdict(set)
dict3 = defaultdict(str)
dict4 = defaultdict(list)
dict1[2] ='two'
 
print(dict1[1])
print(dict2[1])
print(dict3[1])
print(dict4[1])

# 输出值如下：
0
set()
 
[]
```







## 三、python其他

#### 1、.json文件

```python
json文件是一种常见的数据存储文件，比txt看着高级点，比xml看着人性化一点。同时，json作为一种通用协议的文件格式，可以被各种语言方便地读取。所以，json非常适合用来存储结构化的数据。
```

一般情况下的.json文件，存储的是python中的一个dict, python一般直接打开就为字典类型

```json
{
    "name": "dabao",
    "id":123,
    "hobby": {
        "sport": "basketball",
        "book": "python study"
}

```

打开json文件的程序

```python
import json
with open('./test.json') as json_file:
    return json.load(json_file)
```

#### 2、用windows命令行打开pycharm项目

```python
# 1.在当前文件的路径里面输入cmd打开命令行
# 2.在打开的命令行中输入pycharm ./就可以打开当前项目
```



#### 3、python中添加搜索路径的方法

- 方法1：通过sys库来实现

```python
# 方法1:通过前面的sys库来添加搜索路径
import sys
sys.path.append(“路径”)
# 通过这种方法添加的搜索路径只有在运行的时候才可以进行搜索
```

- 方法2：通过.pth来实现

  在anaconda的`D:\Install\anaconda\envs\pytorch1\Lib\site-packages`创建.pth文件，文件名可以是任意的，但是后缀名一定要是.pth文件。

  文件内容可放置路径`D:\matlab\FORCES_PRO` ，注意在这个路径下面用的还是windows自带的路径斜杠。
  
  这样每一次运行就会自动添加搜索路径。

