print("Problem 1: Even or odd->")
print("Write a program that asks the user for a number and prints whether it is even or odd.")

a = float(input("Enter a number:"))

if a%2 == 0:
    print(f"your number {a} is even")
else:
    print(f"your number {a} is odd")


print(50*"-")
print("Problem 2: Guess the Number->")
print("Write a program where the computer stores a secret number and the user tries to guess it")

a = "Hadi$"
while True:
    n = input("Guess:")
    if n == a:
        break
print("your guess was true!", n)

secret = 3010
while True:
    a = int(input("Guess a number:"))
    if a > secret:
        print("guess is high")
    elif a < secret:
        print("guess is low")
    else:
        break
print("you hit the secrert!")

print(50*"-")
print("Problem 3: Simple calculator->")
print("Write a program that asks for two numbers and prints their sum, difference, product, and quotient (if possible).")

def calculator (a,b):
    their_sum = a + b
    their_dif = a - b
    their_product = a*b
    if a%b == 0:
        their_quotient = a/b
    return  (their_sum,their_dif,their_product,their_quotient)

a = int(input("Enter a:"))
b = int(input("Enter b:"))
 
print(f"their sum: {calculator(a,b)[0]},their dif: {calculator(a,b)[1]},their product: {calculator(a,b)[2]},their quotient: {calculator(a,b)[3]}")


print(50*"-")
print("1.Modify the guess-the-number game so the user can guess three times using a loop.")
secret = 20
for i in range(3):
    a = input("guess:")
    if int(a) < secret:
        print("too low")
    elif int(a)> secret:
        print("too high")
    else:
        break
print("you hit it!", int(a))

print(50*"-")
print("2.Write a program that prints the multiplication table (1 to 10) for a given number.")

a = input("enter a number:")
i = 0
while i<10:
    i = i+1
    b = int(a)*i
    print(f"{i} * your choice = {b}")

a = input("enter a number:")
for i in range(1,11):
    print(f"{i} * your choice is {i*int(a)}")


print(50*"-")
print("3.Write a program that asks for five numbers and prints their average.")
i = 0
total = 0
while i<5:
    i = i + 1
    a = input("enter a number:")
    total = total + int(a)
print("mean:",total/5)

total = 0

for i in range(5):
    total += float(input("Enter a number: "))
print("Average:", total / 5)



score = 0
a = input("Enter your name please:")
print(f"Hello {a}, welcome to my adventure game!")

choice_1 = input("you are at a crossroad, choose one option: left, right, stay:")

if choice_1 == "left":
   choice_2 = input("you chose forest! Now tell me climb a tree or follow the path?")
   if choice_2 == "climb a tree":
       score = score + 10
       print("you find the treasure!")
   elif choice_2 == "follow the path":
        score = score + 5
        print("a dog will follow you!")
elif choice_1 == "right":
   choice_2 = input("you chose town! Now tell me library or market?")
   if choice_2 == "library":
       score = score + 10
       print("you will read python!")
   elif choice_2 == "follow the path":
        score = score + 5
        print("you will buy pizza!")
else:
    print("stay here")
print ( " The adventure is over. Thanks for playing, " , a )
print ( " your total sccore is: " , score )

