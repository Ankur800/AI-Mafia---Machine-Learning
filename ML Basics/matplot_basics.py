import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [2, 4, 1]

plt.scatter(x, y)   # scatters the points on graph
plt.plot(x, y)      # draws the line between those points
plt.show()          # shows the the graph/figure

plt.xlabel('x - axis')      # writes some label on x axis
plt.ylabel('y - axis')      # writes some label on y axis
plt.plot(x, y)
plt.title('coding club')    # writes some title on graph
plt.show()

# Line-1 points
x1 = [1, 2, 3]
y1 = [2, 4, 1]

# Line-2 points
x2 = [1, 2, 3]
y2 = [4, 1, 3]

# plotting the line-1 points
plt.plot(x1, y1, label="line 1")

# plotting the line-2 points
plt.plot(x2, y2, label="line 2")

plt.legend()
plt.show()

# Customization of line plot
plt.style.use('fivethirtyeight')
x = [2, 2, 3, 4, 5, 6]
y = [2, 4, 3, 5, 2, 6]

plt.plot(x, y, color='green', linestyle='dashed', linewidth='3',
         marker='o', markerfacecolor='blue', markersize=8)
plt.show()

# BAR CHART
# x-coordinates of left sides of bar
left = [1, 2, 3, 4, 5]

# heights of bars
height = [10, 24, 36, 40, 5]

# labels for bars
tick_labels = ['one', 'two', 'three', 'four', 'five']

# Plotting a bar chart
plt.bar(left, height, tick_label=tick_labels, width=0.8, color=['red', 'blue'])
plt.show()

# HISTOGRAMS
ages = [2, 5, 70, 40, 30, 45, 50, 45, 43, 40, 44,
        60, 7, 13, 57, 18, 90, 77, 32, 21, 20, 40]
# setting the ranges and number of intervals
range = (0, 100)
bins = 10

# Plotting a histogram
plt.hist(ages, bins, range, color='red', histtype='bar', rwidth=0.8)
plt.show()

# PIE CHARTS
# defining labels
activities = ['eat', 'sleep', 'work', 'play']

# portion covered by each label
slices = [3, 7, 8, 6]

# color for each label
colors = ['r', 'm', 'g', 'b']

# Plotting the Pie chart
plt.pie(slices, labels=activities, colors=colors, startangle=90,
        shadow=True, explode=(0, 0, 0.1, 0), radius=1.2, autopct='1.1f%%')
plt.show()
