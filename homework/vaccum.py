import math

def square_vacuum(side):
    room_area = side * side
    coverage = room_area
    ineff = room_area - coverage
    print("\nSquare Vacuum")
    print("Room area:", room_area)
    print("Coverage:", round(coverage, 2))
    print("Inefficiency:", round(ineff, 2))
    print("Efficiency:", round(coverage / room_area * 100, 2), "%")

def circle_vacuum(side):
    room_area = side * side
    r = side / 2
    coverage = math.pi * r * r
    ineff = room_area - coverage
    print("\nCircle Vacuum")
    print("Room area:", room_area)
    print("Coverage:", round(coverage, 2))
    print("Inefficiency:", round(ineff, 2))
    print("Efficiency:", round(coverage / room_area * 100, 2), "%")

def triangle_vacuum(side):
    room_area = side * side
    coverage = (math.sqrt(3) / 4) * (side * side)
    ineff = room_area - coverage
    print("\nTriangle Vacuum")
    print("Room area:", room_area)
    print("Coverage:", round(coverage, 2))
    print("Inefficiency:", round(ineff, 2))
    print("Efficiency:", round(coverage / room_area * 100, 2), "%")

def pentagon_vacuum(side):
    room_area = side * side
    R = side / 2
    coverage = 2.5 * R * R * math.sin(2 * math.pi / 5)
    ineff=room_area-coverage
    print("\nPentagon Vacuum")
    print("Room area:", room_area)
    print("Coverage:", round(coverage, 2))
    print("Inefficiency:", round(ineff, 2))
    print("Efficiency:", round(coverage / room_area * 100, 2), "%")

print("Choose shape: square / circle / triangle / pentagon")
shape = input("Enter shape: ").lower()

side = float(input("Enter room side length: "))

if shape == "square":
    square_vacuum(side)
elif shape == "circle":
    circle_vacuum(side)
elif shape == "triangle":
    triangle_vacuum(side)
elif shape == "pentagon":
    pentagon_vacuum(side)
else:
    print("Invalid shape")
