def linear_model(x, m, c):
    return m * x + c

# Example values
m = 2   # slope
c = 3   # intercept

# Example input values
x_values = [0, 1, 2, 3, 4, 5]

# Calculate the corresponding y values
y_values = [linear_model(x, m, c) for x in x_values]

print("x values:", x_values)
print("y values:", y_values)
