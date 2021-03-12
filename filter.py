# -*- coding: utf-8 -*-
"""
@author: Herikc Brecher
"""
from PIL import Image
import numpy as np
import sys
import time
from tkinter import filedialog
import os 

def gray(convertion_type, img):  
    pixel_set = img.load()
    
    # A list of Pixels is created, to calculate in a simple or weighted way the equivalent in gray scale
    if convertion_type == 'simple':
        for i in range(img.height):
            for j in range(img.width):
                # Calculate the pixel and define the color
                pixel = int(((pixel_set[j,i][0] + pixel_set[j,i][1] + pixel_set[j,i][2]) / 3))
                color = (pixel, pixel, pixel)
                pixel_set[j, i] = color
                
    elif convertion_type == 'weighted':
        for i in range(img.height):
            for j in range(img.width):
                # Calculate the pixel and define the color
                pixel = int(((pixel_set[j,i][0] * 0.299) + (pixel_set[j,i][1] * 0.587) + (pixel_set[j,i][2] * 0.114)))
                color = (pixel, pixel, pixel)
                pixel_set[j, i] = color
    
    return img
                
def negative(img):
    pixel_set = np.asarray(img)
    pixel_set = 255 - pixel_set # Inverts the RGB value
    
    return Image.fromarray(pixel_set)
    
def threshold(img, rgb_above_limiarizacao = 255, rgb_below_limiarizacao = 0, Threshold_Coefficient = 150):  
    # Save the RGB codes of the modified image
    pixel_set = img.load()
    
    # Replaces RGB codes above the coefficient and below, to create regions in the image
    for i in range(img.height):
        for j in range(img.width):
            if pixel_set[j, i][0] > Threshold_Coefficient:
                pixel_set[j, i] = (rgb_above_limiarizacao, rgb_above_limiarizacao, rgb_above_limiarizacao)
            else:
                pixel_set[j, i] = (rgb_below_limiarizacao, rgb_below_limiarizacao, rgb_below_limiarizacao)
                
    return img     

def capture_minus_size(width1, width2, height1, height2):
    temp_size = np.empty(shape=(2)) 
    
    if width1 > width2:
        temp_size[0] = width2
    else:
        temp_size[0] = width1  
        
    if height1 > height2:
        temp_size[1] = height2
    else:
        temp_size[1] = height1  
        
    return temp_size
    
def addition(img1, img2, convertion_type = 'weighted', p1 = 0.5, p2 = 0.5):          
    temp_size = capture_minus_size(img1.width, img2.width, img1.height, img2.height)
    
    # Loads image pixels
    pixels_img1 = img1.load()
    pixels_img2 = img2.load()     
    
    # Create an image 3 to store the final result
    img3 = Image.new("RGB", (int(temp_size[0]), int(temp_size[1])))
    pixels_img3 = img3.load()
    
    if convertion_type == 'simple':
        for i in range(int(temp_size[0])):
            for j in range (int(temp_size[1])):
                # Performs operations to average the R, G and B pixels of image 1 and 2, to store in the new image
                temp = pixels_img1[j,i] + pixels_img2[j,i]
                temp_r = (temp[0] + temp[3]) / 2
                temp_g = (temp[1] + temp[4]) / 2
                temp_b = (temp[2] + temp[5]) / 2
                pixels_img3[j,i] = (int(temp_r), int(temp_g), int(temp_b))
                
    elif convertion_type == 'weighted':
        for i in range(int(temp_size[0])):
            for j in range (int(temp_size[1])):
                # Performs arithmetic operations between the codes R, G and B of image 1 and 2, to store in the new image
                temp = pixels_img1[j,i] + pixels_img2[j,i]
                temp_r = (temp[0] * p1) + (temp[3] * p2)        
                temp_g = (temp[1] * p1) + (temp[4] * p2)     
                temp_b = (temp[2] * p1) + (temp[5] * p2)     
                pixels_img3[j,i] = (int(temp_r), int(temp_g), int(temp_b))    
                
    return img3

def subtraction(img1, img2, convertion_type = 'simple'):          
    temp_size = capture_minus_size(img1.width, img2.width, img1.height, img2.height)
    
    # Loads image pixels
    pixels_img1 = img1.load()
    pixels_img2 = img2.load()     
    
    # Creates an image 3 to store the final result
    img3 = Image.new("RGB", (int(temp_size[0]), int(temp_size[1])))
    pixels_img3 = img3.load()
    
    if convertion_type == 'simple':
        for i in range(int(temp_size[0])):
            for j in range (int(temp_size[1])):
                # Performs arithmetic operations between the codes R, G and B of image 1 and 2, to store in the new image
                temp = pixels_img1[j,i] + pixels_img2[j,i]
                temp_r = (temp[0] - temp[3])
                temp_g = (temp[1] - temp[4])
                temp_b = (temp[2] - temp[5])
                pixels_img3[j,i] = (int(temp_r), int(temp_g), int(temp_b))         
    
    return img3
               
def highlight_edge(img, mask_size = 3): 
    width = img.width
    height = img.height
    
    # Loads pixels in RGB format
    pixel_set = np.asarray(img)

    img2 = Image.new("RGB", (int(width), int(height)))
    pixels_img2 = img2.load()
    
    # Mask creation
    matrix = np.zeros((mask_size, mask_size), dtype=np.int64)
    matrix[0:mask_size, 0] = matrix[0:mask_size, 2] = matrix[0, 1] = matrix[2, 1] = -1
    matrix[1, 1] = 8
    
    for i in range(1, int(width - 2)):
        for j in range (1, int(height - 2)):            
            # Mascara application in the region                   
            new_pixel = matrix * pixel_set[i:i+3, j:j+3, 0]
            
            # Sum of 2x pixels, since a vector is generated in the first
            new_pixel = int(sum(sum(new_pixel)))
            
            # Application of the pixel in the final image
            pixels_img2[j, i] = (new_pixel, new_pixel, new_pixel)
     
    return img2

def dilatation(img, dilatation_value = 255, black_limit = 50):    
    width = img.width
    height = img.height
    
    # Loads pixels in RGB format
    pixel_set = img.load()

    img2 = Image.new("RGB", (int(width), int(height)))
    pixels_img2 = img2.load()
    
    # Mask creation
    matrix = np.zeros((3,3), dtype=np.int64)
    matrix[0:3, 0:3] = dilatation_value
    
    for i in range(int(width - 2)):
        for j in range (int(height - 2)):         
            # Checks if the pixel is considered above the black limit, to expand the mask in the region
            if pixel_set[j, i][0] > black_limit and pixel_set[j, i][1] > black_limit and pixel_set[j, i][2] > black_limit: # R G B 
                # Apply the expansion mask
                for k in range(i - 1, i + matrix.shape[0] - 1):
                    for l in range(j - 1, j + matrix.shape[1] - 1):
                        pixels_img2[l, k] = (dilatation_value, dilatation_value, dilatation_value)
                                                  
    return img2

def erosion(img, original_value = 255, erosion_value = 0, limit = 200):
    width = img.width
    height = img.height

    # Loads pixels in RGB format
    pixel_set = img.load()

    img2 = Image.new("RGB", (int(width), int(height)))
    pixels_img2 = img2.load()
    
    # Mask creation
    matrix = np.zeros((3,3), dtype=np.int64)
    matrix[0:3, 0:3] = erosion_value
    
    for i in range(int(width - 2)):
        for j in range (int(height - 2)):           
            # Checks whether the pixel is considered below the white limit, to erode the mask in the region
            if pixel_set[j, i][0] < limit and pixel_set[j, i][1] < limit and pixel_set[j, i][2] < limit: # R G B 
                
                for k in range(i - 1, i + matrix.shape[0] - 1):
                    for l in range(j - 1, j + matrix.shape[1] - 1):
                        pixels_img2[l, k] = (erosion_value, erosion_value, erosion_value)   
            else:
                pixels_img2[j, i] = pixel_set[j, i]
                      
    return img2

def opening(img, dilatation_value = 255, black_limit = 50, original_value = 255, erosion_value = 0, limit = 200):
    img = erosion(img, original_value, erosion_value, limit)
    img = dilatation(img, dilatation_value, black_limit)
    return img

def closing(img, dilatation_value = 255, black_limit = 50, original_value = 255, erosion_value = 0, limit = 200):
    img = dilatation(img, dilatation_value, black_limit)
    img = erosion(img, original_value, erosion_value, limit)
    return img   

def roberts(img):
    width = img.width
    height = img.height
    
    # Loads pixels in RGB format
    pixel_set = np.asarray(img)

    img2 = Image.new("RGB", (int(width), int(height)))
    pixels_img2 = img2.load()
    
    # Creating the masks
    matrixGx = np.zeros((2, 2), dtype=np.int64)
    matrixGx[0,1] = 1
    matrixGx[1,0] = -1
    
    matrixGy = np.zeros((2, 2), dtype=np.int64)
    matrixGy[0,0] = 1
    matrixGy[1,1] = -1   
    
    for i in range(1, int(width - 1)):
        for j in range (1, int(height - 1)):          
            # Mascara application in the region    
            Gx = matrixGx * pixel_set[i:i+2, j:j+2, 0]
            Gy = matrixGy * pixel_set[i:i+2, j:j+2, 0]
            
            # Sum of 2x pixels, since a vector is generated in the first
            Gx = (sum(sum(Gx))) ** 2
            Gy = (sum(sum(Gy))) ** 2
            
            G = int((Gx + Gy) ** (1/2))
            
            # Application of the pixel in the final image
            pixels_img2[j, i] = (G, G, G)
     
    return img2

def sobel (img):
    width = img.width
    height = img.height
    
    # Loads pixels in RGB format
    pixel_set = np.asarray (img)

    img2 = Image.new ("RGB", (int (width), int (height)))
    pixels_img2 = img2.load ()
    
    # Creating the masks
    matrixGx = np.zeros ((3, 3), dtype = np.int64)
    matrixGx [0,0] = matrixGx [2,0] = 1
    matrixGx [0,2] = matrixGx [2,2] = -1
    matrixGx [1.0] = 2
    matrixGx [1,2] = -2
    
    matrixGy = np.zeros ((3, 3), dtype = np.int64)
    matrixGy [0,0] = matrixGy [0,2] = 1
    matrixGy [2,0] = matrixGy [2,2] = -1
    matrixGy [0.1] = 2
    matrixGy [2.1] = -2
    
    for i in range (1, int (width - 2)):
        for j in range (1, int (height - 2)):
            # Mask application in the region
            Gx = matrixGx * pixel_set [i: i + 3, j: j + 3, 0]
            Gy = matrixGy * pixel_set [i: i + 3, j: j + 3, 0]
            
            # Sum of 2x pixels, since a vector is generated in the first
            Gx = (sum (sum (Gx))) ** 2
            Gy = (sum (sum (Gy))) ** 2
            
            G = int ((Gx + Gy) ** (1/2))
            
            # Application of the pixel in the final image
            pixels_img2 [j + 1, i + 1] = (G, G, G)
            G = 0
     
    return img2

def robinson (img):
    width = img.width
    height = img.height
    
    # Loads pixels in RGB format
    pixel_set = np.asarray (img)

    img2 = Image.new ("RGB", (int (width), int (height)))
    pixels_img2 = img2.load ()
    
    I = []
    # Creating the masks
    matrix1 = np.zeros ((3, 3), dtype = np.int64)
    matrix1 [0,0] = matrix1 [0,2] = 1
    matrix1 [2,0] = matrix1 [2,2] = -1
    matrix1 [0.1] = 2
    matrix1 [2,1] = -2
    I.append (matrix1)
    
    matrix2 = np.zeros ((3, 3), dtype = np.int64)
    matrix2 [1.0] = matrix2 [0.1] = 1
    matrix2 [2,1] = matrix2 [1,2] = -1
    matrix2 [0,0] = 2
    matrix2 [2.2] = -2
    I.append (matrix2)
    
    matrix3 = np.zeros ((3, 3), dtype = np.int64)
    matrix3 [0,0] = matrix3 [2,0] = 1
    matrix3 [0,2] = matrix3 [2,2] = -1
    matrix3 [1.0] = 2
    matrix3 [1,2] = -2
    I.append (matrix3)
    
    matrix4 = np.zeros ((3, 3), dtype = np.int64)
    matrix4 [0.1] = matrix4 [1,2] = -1
    matrix4 [1,0] = matrix4 [2,1] = 1
    matrix4 [0.2] = -2
    matrix4 [2,0] = 2
    I.append (matrix4)
    
    matrix5 = matrix1 * (-1)
    matrix6 = matrix2 * (-1)
    matrix7 = matrix3 * (-1)
    matrix8 = matrix4 * (-1)
    I.append (matrix5)
    I.append (matrix6)
    I.append (matrix7)
    I.append (matrix8)
    
    temp = np.zeros ((8, 3, 3))
    
    for i in range (1, int (width - 2)):
        for j in range (1, int (height - 2)):
            # Mask application in the region
            temp [0:] = I [0:] * pixel_set [i: i + 3, j: j + 3, 0]
            
            G = 0.0
            for p in range (len (I)):
                G = G + (sum (sum (temp [p]))) ** 2
            
            G = int (G ** (1/2))
            
            # Application of the pixel in the final image
            pixels_img2 [j + 1, i + 1] = (G, G, G)
     
    return img2

def menu(option_num):   
    # Window does not appear on the taskbar if used
    #root = tk.Tk()
    #root.withdraw()
    
    filetypes = (("JPG", "*.jpg"), ("PNG", "*.png"), ("JPEG", "*.jpeg"), ("All Files","*.*"))
    title = "Choose a file."
    initDir = os.getcwd()
    defExt = ("All Files","*.*")
    
    print("Warning: Some operations may take several minutes to perform!")
    if (option_num >= 0 and option_num <= 2) or option_num == 5 or (option_num >= 10 and option_num <= 12):
        file_path = filedialog.askopenfilename(initialdir = initDir, filetypes = filetypes, title = title, defaultextension = defExt)
        if file_path == "":
            print("Operação Abortada: Não foi selecionado nenhuma imagem!")
            return
        
        load_img = Image.open(file_path)
    elif option_num == 3 or option_num == 4:
        print("It is necessary to choose two images for this operation")
        file_path = filedialog.askopenfilenames(initialdir=initDir, filetypes = filetypes, title = title, defaultextension = defExt)
        if len(file_path) != 2:
            print("Operation Aborted: The correct number of images has not been selected!")
            return
        load_imgs1 = Image.open(file_path[0])
        load_imgs2 = Image.open(file_path[1])
    elif option_num >= 6 and option_num <= 9:
        print("It is necessary to choose a black and white image for this operation")
        file_path = filedialog.askopenfilename(initialdir = initDir, filetypes = filetypes, title = title, defaultextension = defExt)
        if file_path == "":
            print("Operation Aborted: No image selected!")
            return
        
        load_img = Image.open(file_path)
          
    
    if option_num == 0:          
        while True:
            gray_type = str (input ("Do you want to perform simple or weighted conversion:"))
            if gray_type == 'simple' or gray_type == 'weighted':
                break
            else:
                print ("Invalid Option")
                
        start = time.time()  
        img = gray(gray_type, load_img)
        img.save("new_images/gray.png")
    
    elif option_num == 1:    
        start = time.time()   
        img = negative(load_img)
        img.save("new_images/negative.png")
    
    elif option_num == 2:      
        start = time.time()   
        gray_img = gray('weighted', load_img)
        
        img = threshold(gray_img, 255, 0, 150)
        img.save("new_images/threshold.png")
    
    elif option_num == 3:  
        while True:
            add_type = str (input ("Do you want to perform simple or weighted addition:"))
            if add_type == 'simple' or add_type == 'weighted':
                break
            else:
                print ("Invalid Option")
        
        p1 = 0.0
        p2 = 0.0
        if add_type == 'weighted':
            print("Warning: 1 is 100% and 0 is 0%, 0.5 is 50%.")
            p1 = float(input("How many percent do you want image 1 to represent?"))
            p2 = float(input("How many percent do you want image 2 to represent?"))
        
        start = time.time()   
        img = addition(load_imgs1, load_imgs2, add_type, p1 = p1, p2 = p2)               
        img.save("new_images/addition.png")                                                                        
        
    elif option_num == 4:
        start = time.time()   
        img = subtraction(load_imgs1, load_imgs2)               
        img.save("new_images/subtraction.png")   
        
    elif option_num == 5:
        start = time.time()   
        gray_img = gray('weighted', load_img)
        
        img = highlight_edge(gray_img, 3)
        img.save("new_images/highlight_edge.png")  
          
    elif option_num == 6:
        start = time.time()   
        img = dilatation(load_img)
        img.save("new_images/dilatation.png")    
          
    elif option_num == 7:
        start = time.time()   
        img = erosion(load_img)
        img.save("new_images/erosion.png")        

    elif option_num == 8:
        start = time.time()   
        img = opening(load_img)
        img.save("new_images/opening.png")   

    elif option_num == 9:
        start = time.time()   
        img = closing(load_img)
        img.save("new_images/closing.png")      

    elif option_num == 10:
        start = time.time()   
        gray_img = gray('weighted', load_img)
        
        img = roberts(gray_img)
        img.save("new_images/roberts.png")    

    elif option_num == 11:
        start = time.time()   
        gray_img = gray('weighted', load_img)
        
        img = sobel(gray_img)
        img.save("new_images/sobel.png") 
        
    elif option_num == 12:
        start = time.time()   
        gray_img = gray('weighted', load_img)
        
        img = robinson(gray_img)
        img.save("new_images/robinson.png")    
                                 
    end = time.time()
    print("Runtime:% .2f seconds"% round (end - start, 2))
    img.show()  

def print_menu ():
    print("Warning: When selecting the image, after the menu, check that the dialog window is not open using ALT + TAB.")
    print("0 - Grayscale")
    print("1 - Negative")
    print("2 - Threshold")
    print("3 - Addition")
    print("4 - Subtraction")
    print("5 - Edge Highlight")
    print("6 - Expansion")
    print("7 - Erosion")
    print("8 - Aperture")
    print("9 - Closing")
    print("10 - Roberts")
    print("11 - Sobel")
    print("12 - Robinson")
    print("13 - Exit")
    
    while True:
        option = int (input ("Choose an option:"))
        if option >= 0 and option <= 13:
            break
        else:
            print ("Invalid Option")
    
    return option

















