import math

def p5(image): #return edge_image

    height = len(image)
    width = len(image[0])
    ret_img = [[0 for x in range(width)] for y in range(height)]
    sobel_33_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_33_y = [[1,2,1],[0,0,0],[-1,-2,-1]]


    def convolution(input_1, input_2):  # input_1 is the larger matrix
        height = len(input_1)
        width = len(input_1[0])
        a = len(input_2)
        b = len(input_2[0])
        result = [[0 for x in range(width)] for y in range(height)]

        for i in range(height):
            for j in range(width):
                for k in range(a):
                    for l in range(b):
                        result[i][j] += input_2[k][l] * input_1[i - k][j - l]
        return result
    gradient_x = convolution(image, sobel_33_x)
    gradient_y = convolution(image, sobel_33_y)

    for i in range(height):
        for j in range(width):
            ret_img[i][j] = math.sqrt(gradient_x[i][j]**2 + gradient_y[i][j]**2)
            if i == 0 or j == 0 or i == height - 1 or j == height - 1:
                ret_img[i][j] = 0
    return ret_img # use plt.imshow() to show it



