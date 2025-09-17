#第一题
import turtle
t = turtle.Turtle()
for sj in range(3): #三角形
    t.forward(100)
    t.left(120)
for zf in range(4): #正方形
    t.forward(100)
    t.right(90)
for lb in range(6): #六边形
    t.left(60)
    t.forward(100)
turtle.done()

#第二题
import turtle
y = turtle.Turtle()
i = 50
for txy in range(5): #同心圆
    y.circle(i)
    y.penup()
    y.goto(0,-20*(txy+1))
    y.pendown()
    i += 20
turtle.done()

#第三题
import turtle
wjx = turtle.Turtle()
wjx.fillcolor("yellow")
wjx.begin_fill()
for wj in range(5): #五角星
    wjx.forward(200)
    wjx.right(144)
wjx.end_fill()
turtle.done()
