import matplotlib.pyplot as plt

# Data for x-axis and y-axis
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Plot the data on the axis
plt.plot(x, y)

# Set labels for x-axis and y-axis
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Set title for the plot
plt.title('Line Graph Example')

## Display the plot
#plt.show()

#Save plot
plt.savefig('basic.png')
