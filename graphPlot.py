import matplotlib.pyplot as plt

# x axis values
x = [31,41,51,61,71,81,91,101,111]

# corresponding y axis values
y = [0.09404826164245605,0.33559703826904297,0.9701879024505615,4.295067071914673,9.976176261901855,18.92745542526245,60.08960580825806,102.63305878639221,150.85634446144104]

# plotting the points
plt.plot(x, y,label = 'N',marker = 'o',color = 'red', linestyle = '-', markersize = 6)

# naming the x axis
plt.xlabel('N(size of polynomial)')

# naming the y axis
plt.ylabel('Time(seconds)')

# giving a title to my graph
plt.title('Time vs N on intel i5 processor')

# function to show the plot
plt.show()


