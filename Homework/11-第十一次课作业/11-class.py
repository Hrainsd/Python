#Turtle库
import turtle #三角形
t = turtle.Turtle()
t.forward(100) #foreward表示向前走，可以用fd代替foreward
t.left(120) #left表示左转
t.forward(100)
t.left(120)
t.forward(100)
turtle.done()

import turtle #正方形1
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.left(90)
turtle.forward(100)
turtle.done()

import turtle #正方形2
t = turtle.Turtle()
for i in range(4):
    t.forward(100)
    t.right(90)
turtle.done()

import turtle #圆
t = turtle.Turtle()
t.circle(50) #turtle.circle(r,extent)画（半径，角度）的圆
turtle.done()

#from 库名 import 函数名
#from 库名 import* 函数名
#turtle.setup(宽度,高度,距框左边的距离,距框上边的距离)
import turtle #五角星
turtle.goto(-100,100) #(x,y)坐标系，默认从（0，0）开始
turtle.goto(100,100)
turtle.goto(-100,-100)
turtle.goto(0,200)
turtle.goto(100,-100)
turtle.goto(-100,100)

turtle.color()
turtle.colormode() #1.0表示RGB小数值模式，255表示RGB整数值模式
turtle.penup() #抬起画笔
turtle.pendown() #落下画笔
turtle.pensize() #画笔的宽度
turtle.pencolor() #画笔的颜色，颜色字符串或者R,G,B值
turtle.setheading() #改变前进的绝对角度
import turtle
turtle.pencolor("purple")
turtle.pencolor(0.63,0.13,0.94)
turtle.circle(100)
turtle.circle(-100,90)
