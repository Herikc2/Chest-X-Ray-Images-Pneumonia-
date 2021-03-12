# -*- coding: utf-8 -*-
"""
@author: Herikc Brecher
@Description: Library to take statistics about the dataframe
"""

# Importing libraries
import pandas as pd 
import scipy.stats as scipy
import seaborn as sns
import statistics as stat
import matplotlib.pyplot as plt

def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1  
    return df.sort_index()  

def main(x):
     
    '''
        A histogram is created to analyze the distribution of the images
        The X axis represents the quantity
        The Y axis represents the Width or Height
        
        NOTICE: To perform it is necessary to receive the dataset without normalization
    '''
    
    dataframe = pd.DataFrame(columns = ['ID', 'Width', 'Height'])

    for i in range(len(x)):
        add_row(dataframe, [i, x[i].shape[0], x[i].shape[1]])
        
    dataframe = dataframe.astype({"Width": int, "Height": int})
    
    # Seaborn histograms with normal curve
    sns.distplot(dataframe['Width'], kde = True, rug = False, hist = True, bins = 10, axlabel = "Width")
    plt.savefig('histogram_width_normal.png', format='png')
    plt.show()
    sns.distplot(dataframe['Height'], kde = True, rug = False, hist = True, bins = 10, axlabel = "Height")
    plt.savefig('histogram_height_normal.png', format='png')
    plt.show()
    
    # Matplotlib histograms by frequency
    dataframe.hist (column = 'Width', bins = 10)
    plt.xlabel('Frequency')
    plt.ylabel('Width')
    plt.title('Distribution on Width in Images')
    plt.savefig('histogram_width.png', format = 'png')
    plt.show()
    
    dataframe.hist (column = 'Height', bins = 10)
    plt.xlabel('Frequency')
    plt.ylabel('Height')
    plt.title('Distribution on Height in Images')
    plt.savefig('histogram_height.png', format = 'png')
    plt.show()
      
    # Central measures
    max_height          = max(dataframe['Height'])
    max_width           = max(dataframe['Width'])
    mean_height        = stat.mean(dataframe['Height'])
    mean_width         = stat.mean(dataframe['Width'])
    min_height          = min(dataframe['Height'])
    min_width           = min(dataframe['Width'])
    median_height      = stat.median(dataframe['Height'])
    median_width       = stat.median(dataframe['Width'])
    
    # Standard deviation and variance
    deviation_height       = stat.pstdev(dataframe['Height'])
    deviation_width        = stat.pstdev(dataframe['Width'])
    variance_height    = stat.pvariance(dataframe['Height'])
    variance_width     = stat.pvariance(dataframe['Width'])
    
    print("Maximum height:", max_height)
    print("Maximum width:", max_width, "\ n")
    print("Minimum height:", min_height)
    print("Minimum width:", min_width, "\ n")
    print("Media Height:", mean_height)
    print("Media Width:", mean_width, "\ n")
    print("Median Height:", median_height)
    print("Median Width:", median_width, "\ n")
    
    print("Standard Deviation Height:", deviation_height)
    print("Standard Deviation Width:", deviation_width, "\ n")
    print("Variancia Height:", variance_height)
    print("Variancia Width:", variance_width, "\ n")
    
    
    '''
        # QUARTIS
    
        Quartiles (Q1, Q2 and Q3): These are values ​​given from the set of observations ordered in ascending order,
        that divide the distribution into four equal parts. The first quartile, Q1, is the number that leaves 25% of
        observations below and 75% above, while the third quartile, Q3, leaves 75% of the observations below and
        25% above. Q2 is the median, leaving 50% of the observations below and 50% of the observations above.
        
        Reference: http://www.portalaction.com.br/estatistica-basica/23-quartis
    '''
    
    first_quatile = dataframe.quantile(q=0.25, axis=0, numeric_only=True, interpolation='linear')
    second_quatile = dataframe.quantile(q=0.50, axis=0, numeric_only=True, interpolation='linear')
    third_quartile = dataframe.quantile(q=0.75, axis=0, numeric_only=True, interpolation='linear')
    fourth_quatile = dataframe.quantile(q=1.00, axis=0, numeric_only=True, interpolation='linear')
    
    print ("First Quartile Height:", first_quatile[1])
    print ("First Quartile Width:", first_quatile[0], "\ n")
    
    print ("Second Quartile Height (Median):", second_quatile[1])
    print ("Second Quartile Width (Median):", second_quatile[0], "\ n")
    
    print ("Third Quartile Height:", third_quartile[1])
    print ("Third Quartile Width:", third_quartile[0], "\ n")
    
    print ("Quartile Height:", fourth_quatile[1])
    print ("Fourth Quartile Width:", fourth_quatile[0], "\ n")
    
    # BOX PLOT 
    
    sns.set(style="whitegrid", color_codes=True)
    
    sns.boxplot(data = dataframe['Height']).set_title("Box Plot Height")
    plt.savefig('box_plot_height.png', format='png')
    plt.show()
    
    sns.boxplot(data = dataframe['Width']).set_title("Box Plot Width")
    plt.savefig('box_plot_width.png', format='png')
    plt.show()
    
    '''
        # Asymmetry coefficient
        
        Asymmetry where 0 characterizes symmetry, greater than 0 characterizes greater right-hand distribution
        and less than 0 characterizes a greater distribution on the left.
    '''
    
    asymmetry_coefficient = dataframe.skew()
    print("Height Asymmetry Coefficient:", asymmetry_coefficient [2])
    print("Asymmetry Coefficient Width:", asymmetry_coefficient [1], "\ n")
    
    '''
        # Kurtosis
        
        If b2 = 0, then the distribution function has the same flatness as the normal distribution, we call these functions
        of mesocurtics.
        
        If b2> 0, we say that the distribution function is leptocurtic and has the distribution function curve more
        tapered with a peak higher than the normal distribution.
        
        If b2 <0, then the distribution function is more flattened than the normal distribution. We say that this curve of
        distribution function is platicurtic.
        Reference: http://www.portalaction.com.br/estatistica-basica/26-curtose
    '''
    
    curtose = scipy.kurtosis(dataframe)
    print("Curtose Height:", curtose[2])
    print("Curtose Width:", curtose[1], "\n")






