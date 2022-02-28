import matplotlib.pyplot as plt

vgg16_list = [24.35, 39.58, 49.58, 60.48, 66.39, 71.68, 72.89, 74.74, 75.69, 74.80, 76.08, 76.93, 75.87, 77.05, 76.70, 76.01, 77.76, 75.08, 76.85, 77.39, 76.57, 77.67, 77.71, 77.18, 77.72, 76.57, 76.54, 78.11, 77.65, 76.77, 78.41, 78.37, 77.20, 77.59, 76.91, 76.89, 78.07, 78.00, 77.54, 78.00, 77.70, 77.41, 74.76, 77.50, 78.79, 78.60, 78.74, 78.39, 78.26, 78.44, 77.09, 78.28, 78.20, 77.80, 78.66, 78.38, 77.72, 77.83, 78.09, 76.81, 77.43, 78.48, 78.12, 78.59, 78.94, 77.97, 77.20, 76.72, 77.88, 78.77, 78.86, 78.03, 77.45, 78.03, 78.46, 78.79, 77.98, 78.62, 79.14, 76.20, 79.23, 78.16, 79.15, 79.46, 78.74, 78.16, 74.75, 78.78, 79.96, 79.06, 78.71, 78.36, 79.53, 79.14, 79.11, 78.76, 77.42, 78.88, 77.05, 79.59]
vgg11_list = [39.43, 59.80, 66.52, 69.84, 72.64, 72.68, 73.16, 76.25, 74.23, 74.75, 74.67, 75.72, 75.66, 75.59, 75.29, 75.67, 75.99, 75.41, 74.85, 75.24, 75.43, 75.23, 75.09, 75.52, 74.80, 75.51, 75.51, 75.16, 75.40, 75.72, 75.56, 75.98, 75.57, 75.33, 75.87, 76.12, 76.18, 75.61, 75.63, 74.74, 75.97, 75.70, 76.30, 76.26, 74.93, 75.53, 75.55, 75.23, 76.81, 75.50, 75.84, 74.42, 76.26, 76.07, 75.45, 75.99, 75.65, 75.68, 75.76, 75.35, 75.55, 76.43, 75.96, 75.06, 76.13, 75.16, 75.98, 76.09, 76.26, 76.04, 76.39, 76.01, 77.23, 75.87, 76.26, 75.87, 76.26, 74.75, 75.81, 76.40, 76.35, 76.66, 75.94, 76.06, 77.12, 76.01, 75.59, 76.90, 75.92, 75.68, 76.43, 76.78, 75.08, 76.15, 75.79, 76.81, 75.97, 73.30, 76.08, 74.48 ]

plt.plot(vgg11_list, 'g', label='VGG11 Accuracy')
plt.plot(vgg16_list, 'b', label='VGG11 Accuracy')
plt.title('Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy[%]')
plt.legend()
plt.grid()
plt.savefig("VGG11and16_Test_Accuracy_plot.png")