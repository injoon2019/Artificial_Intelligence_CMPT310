# -*- coding: utf-8 -*-
"""
Created on Wed May 13 03:22:00 2020

@author: Injun Son
"""

"""
Installing Anaconda for Python 3.5 or above:
https://docs.anaconda.com/anaconda/install/

Alternatively, you can download and install python from:
https://www.python.org/downloads/


Python IDE (Integrated Development Environment):
Spyder, Pycharm, Visual Studio Code (VS code), etc.


Clone or download AIMA Python and install the requirements:
https://github.com/aimacode/aima-python
If you have installed Anaconda, you may install the AIMA Python requirements
using Anaconda Prompt on Windows or terminal on Linux (to find Anaconda Prompt
on Windows, from the Start menu, search for and open "Anaconda Prompt"):
cd aima-python
pip install -r requirements.txt
"""
import sys
sys.path.insert(0, 'C:\\Users\\injoo\\aima-python')

from search import * # This should run without error messages

# Python - Variable Types:

sum_number = 100    #An integer number
height = 177.0      #floating point number
name = "Injun"      #A string
course = 'CMPT 310' #A string
print(name)
print(course)
print(name+ ' ' +course)    #Injun CMPT 310
print(name[2:4])    #nj
print(len(name))    #5
del course          #delete

#Python Lists:

numbers = [1, 3, 12, 9, -2, 4]
print(numbers[1:4])     #[3, 12, 9]
print(numbers[1:])      #[3, 12, 9, -2, 4]
print(numbers[-1])      # 4
print(numbers[-3:])     #[9, -2, 4]

listA = [3.1, 'b', 'a', 2, [4,5,6]]
print(listA[3]) #2
print(listA[4]) # [4,5,6]
listB = [-4.3, 3.5, -2.2, 7.7, -1.1, 5.5]
print(listB.index(7.7)) # 3
listB.append(4.4)   
print(listB) #[-4.3, 3.5, -2.2, 7.7, -1.1, 5.5, 4.4]
listB.sort()
print(listB) #[-4.3, -2.2, -1.1, 3.5, 4.4, 5.5, 7.7]
x = listB.pop()
print(x) # 7.7
listC = ['c', 'k', 'g', 'b']
listD = sorted(listC)
print(listC) # ['c', 'k', 'g', 'b']
print(listD) # ['b', 'c', 'g', 'k']
listD[2] = 'r'
print(listD) # ['b', 'c', 'r', 'k']

# Python Tuples:

t = (2, 'a', 1.3)
#Tuples are immutable:
# t[0] = 4 # TypeError: 'tuple' object does not support item assignment 

# Python Sets:
a = set()   # Create an empty set
a.add(2)
a.add(3)
a.add(2)
print(a) #{2,3}

# Python Dictionaries:

dictA =  {}
dictA['a'] = 1
dictA['b'] = 2
dictA['c'] = 3
print(dictA) # {'a':1, 'b':2, 'c':3}
print(dictA.keys()) # dict_keys(['a', 'b', 'c'])
print(dictA.values())   # dict_values([1, 2, 3])

for x in listB:
    print(x)

for i in range(6):
    print(i)

for i in range(len(listB)):
    print(listB[i])

for x in listB:
    if x>0:
        print(x)
    else:
        print("{:.2f}".format(x**3))

for x in dictA:
    print(x)
for x in dictA:
    print(x, end=' ') # a b c
print('\n')

for y in dictA.values():
    print(y)
for z in dictA.items():
    print(z)
for x, y in dictA.items():
    print((x,y))

dictA_values = [x for x in dictA.values()]
print(dictA_values) # [1, 2, 3]

listE = [(y,x) for x,y in dictA.items()]
print(listE)    # [{1: 'a'}, {2: 'b'}, {3: 'c'}]

dictB = {}
for x, y in dictA.items():
    dictB[y] = x
print(dictB) # {1: 'a', 2: 'b', 3: 'c'}

'''
https://medium.com/@tyastropheus/tricky-python-i-memory-management-for-mutable-
immutable-objects-21507d1e5b95
The value of a mutable object can be modified in place after it’s creation,
while the value of an immutable object cannot be changed.:
Immutable Objects: int, float, long, complex, string tuple, bool
Mutable Objects: list, dict, set, byte array, user-defined classes
'''
listA = [1,2,3,4]
listB = listA
print(id(listA)) #2546043846792
print(id(listB)) #2546043846792
listA.append(5)
print(listB)    #[1, 2, 3, 4, 5]
print(id(listA)) #1352999960968

x = 5
y = x
print(id(x))    #140704380002832
print(id(y))    #140704380002832
x+=1
print(x)
print(y)
print(id(x))
print(id(y))

# Python Functions:

def my_function(x: float) -> float:
    return x*x

print(my_function(3.0)) 

# Problem: Give an array of numbers find the mode of the numbers.
# The mode of an array of numbers is the number that occurs most frequenrly in the array

def find_mode(listA: list)-> float:
    if len(listA)==0:
        return None
    dictA = {}  #keys are numbers and values are freqeuncies
    max_fr = 1 #Max frequency so far
    mode = listA[0] # Mode so far
    for x in listA:
        if x in dictA:  #The complexity of this line in O(1)
            dictA[x] +=1
            if dictA[x] +1 > max_fr:
                max_fr = dictA[x]+1
                mode = x
        else:
            dictA[x] =1
    return mode

print(find_mode([8.1, -1.2, 3.4, -1.2, -2, -1.2, 8.1, 0.0, 3.4, -3.0, -2]))

mode = find_mode([])
print(mode)

import numpy
numbers = list(numpy.random.randint(low=0, high=100, size=100000))
import time
t_0 = time.clock()
mode = find_mode(numbers)
t_1 = time.clock()
print(mode)
print("Time elapsed: ", t_1 - t_0)
del numbers

# The complexity of the above algorithm is in O(n), where n is the length of
# the input array. This is the best algorithm you can find for this problem!

# Problem: Write a function that returns n'th number in the Fibonacci series:
# the sum of two elements defines the next

def fib(n: int )-> int:
    if n==0:
        return 0
    if n==1:
        return 1
    a, b = 0, 1
    m = 0
    while m< n:
        a, b = b, a+b
        m += 1
    return a

fib_numbers = [fib(i) for i in range(15)]
print(fib_numbers) # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

# Problem: Write a recursive function that returns the reverse of an input 
# string. For example, the reverse of "babbcde" is "edcbbab".
def rev(input_st: str) -> str:
    if(len(input_st) < 1):
        return input_st
    return rev(input_st[1:]) + input_st[0]
print(rev('bcbbcde'))

#Data Type Conversion:
x='11'
#x = (input("Please enter a number: ")) # 11 Enter
print(int(x)) # 11
print(float(x)) # 11.0

strA = "This is a CMPT 310 tutorial"
print(strA.lower()) #this is a cmpt 310 tutorial
strA_list = strA.split()
print(strA_list) # 

numList = ['1', '2', '3', '4']
separator = '@'
strB = separator.join(numList)
print(strB) #1@2@3@4

def bin_search(a: list, x: float) -> int:
        l = 0
        r = len(a) -1
        while l<=r:
            m = (l+r) //2
            if a[m] == x:
                return m
            if a[m] > x:
                r = m -1
            else:
                l = m + 1
        return -1
listA = sorted([1.2, -1.2, 3.0, 0.0, 3.4, 5.0, 2.0, 4.0, 4.4])
print(listA) # [-1.2, 0.0, 1.2, 2.0, 3.0, 3.4, 4.0, 4.4, 5.0]
print(bin_search(listA, 4.4)) # 7

import math
class Complex:
    '''
    This is a class for complex numbers:
    https://en.wikipedia.org/wiki/Complex_number
    '''
    def __init__(self, real, imag):
        self.x = real
        self.y = imag
    def conjugate(self):
        return Complex(self.x, -self.y)
    def modulus(self):
        return math.sqrt(self.x * self.x + self.y* self.y)
    def reciprocal(self):
        r = self.modulus()* self.modulus()
        assert r>0, "The reciprocal of (0,0) is not defined"
        return Complex(self.x / r, -self.y/r)
    def print_complex(self)
        print((self.x, self.y))
def add_complex(z1: Complex, z2: Complex):
    return Complex(z1.x+z2.x, z1.y+z2.y)

    z = Complex(3.0, -4.5) # z  = (3.0, -4.5) = 3.0 - 4.5 i
                  
z.print_complex()   # (3.0, -4.5)  
w = z.conjugate()
w.print_complex()  # (3.0, 4.5)
q = z.reciprocal()
q.print_complex() # (0.10256410256410256, 0.15384615384615385)
print(Complex.__doc__)
q = add_complex(z, w)
q.print_complex() # (6.0, 0.0)


class BSTNode:
    def __init__(self, data):
        self.right = None
        self.left = None
        self.data = data

numpy.random.seed(12345)
a = []
for i in range(10):
    #a.append(numpy.random.randint(0, 1000))
    a.append(numpy.random.uniform(0,1))

def addNode(root, k):
    temp = BSTNode(k)
    if root == None:
        return temp

    x =root
    while x != None:
        y = x
        if k >= x.data:
            x = x.right
        else:
            x = x.left
    if (k>= y.data):
        y.right = temp
    else:
        y.left = temp
    return root

root = None
for x in a:
    root = addNode(root, x)

def inorder(root):
    if (root != None):
        inorder(root.left)
        print(root.data)
        inorder(root.righit)

inorder(root)

def postorder(root):
    if( root!= None):
        postorder(root.left)
        postorder(root.right)
        print(root.data)

def height(root):
    if not root:
        return 0
    return max(height(root.left), height(root.right))+1
print(height(root))

def diameter_height(root):
    if not root:
        return (0,0)
    (left_d, left_h) = diameter_height(root.left)
    (right_d, right_h) = diameter_height(root.right)
    return (max(left_d, right_d, left_h + right_h + 1), max(left_h, right_h) + 1)

print("Diameter and Height:"),
print(diameter_height(root))

def total_sum_all_root_to_leave_paths(root, current_sum, total_sum):
    if not root:
        return
    if not root.right and not root:
        total_sum[0] += current_sum + root.data
    total_sum_all_root_to_leave_paths(root.left, current_sum + root.data, total_sum)
    total_sum_all_root_to_leave_paths(root.right, current_sum + root.data, total_sum)
    

total_sum = [0.0]
total_sum_all_root_to_leave_paths(root, 0.0, total_sum)
print(total_sum)


import heapq
def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]

print(heapsort([1,3,5,7,9,2,4,6,8,0]))

stack =[]
stack.append('a')
stack.append('b')
stack.append('c')
print(stack)
x = stack.pop()
print(x) #'c'

from collections import deque
qu = deque(["Eric", "John", "Michael"])
qu.append("Terry")
x = qu.popleft()
print(x) #Eric