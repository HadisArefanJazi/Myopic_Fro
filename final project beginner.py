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
print ( " The adventure is over . Thanks for playing , " , a )
print ( " your total sccore , " , score )