#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:57:47 2023

@author: spayet

TP 9 for MEALOR II
"""

import math 
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

def find_closest_points(x,y,x_candidates, y_candidates,nb_points,values):
    '''
    Find the nb_points closest points to a point of coordinates (x,y) among
    a certain number of candidates and provide the associated value

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates of the point
    x_candidates, y_candidates: array-like, 1d
        x and y coordinates of candidate points
    nb_points: int
        number of closest points to select. 

    Returns
    -------
    Return 2 lists of coordinates x_selected,y_selected

    '''
    x_selected = []
    y_selected =[]
    values_selected = []
    distances = []
    for index_point in np.arange(0,len(x_candidates)):
        distance = math.sqrt(pow(x-x_candidates[index_point],2)+
                             pow(y-y_candidates[index_point],2))
        distances.append(distance)
    closest_points_indices = np.argsort(distances)
    #print("closest_points_indices with function",closest_points_indices) 

    for i in np.arange(0,nb_points):
        x_selected.append(x_candidates[closest_points_indices[i]])
        y_selected.append(y_candidates[closest_points_indices[i]])
        values_selected.append(values[closest_points_indices[i]])
    return x_selected, y_selected, values_selected


def second_order_polynomial_approximation(x, y, x_sample, y_sample, data):
    '''
    Polynomial approximation by least squares fitting.
    Polynom is a + b x + c y + d x**2 + e y**2 + f x y

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates of the point where data must be estimated.
    x_sample, y_sample: array-like, 1d
        x and y coordinates of points where data are known.
    data: np.ndarray, 2d
        Surface to fit. 

    Returns
    -------
    Return value at coordinates (x,y)

    '''

    #Compute centered coordinates
    X_sample_centered = []
    Y_sample_centered = []
    for index in np.arange(0,len(x_sample)):
        X_sample_centered.append(x_sample[index]-x)
        Y_sample_centered.append(y_sample[index]-y)
    #Compute A-matrix and B-vector (could have used weights)
    A = np.zeros((6,6))
    B = np.zeros((6,1))
    P = np.ones((1,6))
    for i in np.arange(0,len(x_sample)):
        P[0,1] = X_sample_centered[i]
        P[0,2] = Y_sample_centered[i]
        P[0,3] = X_sample_centered[i]**2
        P[0,4] = Y_sample_centered[i]**2
        P[0,5] = X_sample_centered[i]*Y_sample_centered[i]
        B = B + data[i]*P.T
        A = A + P.T*P
    sol, res, rank , s = np.linalg.lstsq(A, B)
    return sol[0][0] #value at the center


# %% ======================================================================
""" 
Initialization
"""
# =========================================================================

time = []
X_position = []
Y_position = []
damage = []
X_position_with_high_damage = []
Y_position_with_high_damage = []
high_damage = []
X_all_segments = []
Y_all_segments = []

# %% ======================================================================
""" 
Read integ result file, such as
...
#  elem  ip  X  Y  epcum_bar_max  wpmax  
4650  1  2.6675824833e+00  2.1360321500e-01  0.0000000000e+00  0.0000000000e+00  
4650  2  2.6944434333e+00  1.5180975000e-01  0.0000000000e+00  0.0000000000e+00  
4650  3  2.7951194833e+00  1.7468474500e-01  0.0000000000e+00  0.0000000000e+00  
# ===  time  3.0000000000e-02
#  elem  ip  X  Y  epcum_bar_max  wpmax  
1  1  2.0374861000e+00  8.0624543167e+00  0.0000000000e+00  0.0000000000e+00  
1  2  2.0541407500e+00  7.9675034167e+00  0.0000000000e+00  0.0000000000e+00  
1  3  2.1614351500e+00  8.0264306667e+00  0.0000000000e+00  0.0000000000e+00 
...
"""
# =========================================================================

selected_time = 0.50 
file = open('var.post')
lines = file.readlines()
file.close()
for line in lines:
    text_list = line.split()
    if line.startswith('#'):
        if text_list[2] == 'time':
            time.append(float(text_list[-1]))
    else :
        if time[-1] == selected_time :
            #Get the coordinates in the reference configuration
            X_position.append(float(text_list[2]))
            Y_position.append(float(text_list[3]))
            damage.append(float(text_list[5]))

print("Data loaded for time = ",time)
            
# %% ======================================================================
""" Visualization """
# =========================================================================

plt.scatter(X_position, Y_position, c=damage, cmap='jet', marker = ".")
plt.axis("equal")
plt.colorbar(orientation="vertical")
plt.title('Damage map for time %f'%selected_time)
plt.show()  

# %% ======================================================================
""" Select only the higly damaged areas """
# =========================================================================

"""
#Damage value to distinguish higly damaged ares
threshold_damage = ...#value between 0 and 1
index = np.where(np.array(damage) > threshold_damage)
for i in index[0]:
    X_position_with_high_damage.append(X_position[i])
    Y_position_with_high_damage.append(Y_position[i])
    high_damage.append(damage[i])
    
plt.scatter(X_position_with_high_damage, Y_position_with_high_damage, c=high_damage, cmap='jet', marker = ".")
plt.axis("equal")
plt.colorbar(orientation="vertical")
plt.title('Points with damage greater than %f'%threshold_damage)
plt.show()
"""

# %%===================================================
'''Use DBSCAN to separate the damaged areas'''
# =====================================================
"""
# DBSCAN parameters
# The maximum distance between two samples for one to be considered as in the
# neighborhood of the other. This is not a maximum bound on the distances of
# points within a cluster. This is the most important DBSCAN parameter to
# choose appropriately for your data set and distance function. default=5
EPS_DBSCAN = 5
# The number of samples (or total weight) in a neighborhood for a point to be
# considered as a core point. This includes the point itself.
MIN_SAMPLES_DBSCAN = 10
# Concatenation to fit the FORMAT required by DBSCAN
data_set = np.dstack((X_position_with_high_damage, Y_position_with_high_damage))
X = data_set[0]
if len(X)>0:
    # use DBSCAN (sklearn.cluster.DBSCAN) to find clusters
    db = DBSCAN(eps=EPS_DBSCAN, min_samples=MIN_SAMPLES_DBSCAN).fit(X)
    # Count number of clouds and points in cloud and corresponding to noise
    # label -1 corresponds to noise
    labels = db.labels_
    print("labels", labels)
"""
# %%===================================================
'''Consider each damaged area independently'''
# =====================================================
"""
for label in np.arange(0, max(labels)+1):
    local_X_position = []
    local_Y_position = []
    local_damage = []
    indices = np.where(labels == label)
    print("indices",indices)
    for i in indices[0]:
        local_X_position.append(...)
        local_Y_position.append(...)
        local_damage.append(...)
""" 
"""     
    # %%===================================================
    '''Find the point with maximum damage'''
    # =====================================================
    
    ...
    X_max_damage = ...
    Y_max_damage = ...
    
    plt.scatter(local_X_position, local_Y_position, c=local_damage, cmap="jet")
    plt.colorbar(orientation="vertical")
    plt.title("Damage map for area with label %d"%(label))
    plt.scatter(X_max_damage, Y_max_damage, color = "black", marker="*", s=100,
                label='maximum damage')
    plt.axis("equal")
    plt.legend()
    plt.show() 
"""
"""
    # %%===================================================
    '''Position evaluation points on a circle'''
    # =====================================================
    
    radius = 0.15
    X_evaluation_point_position = []
    Y_evaluation_point_position = []
    angular_accuracy = 5
    evaluation_angles = np.arange(0,360,angular_accuracy)
    for theta_index in evaluation_angles :
        theta = float(theta_index) * math.pi /180. 
        X_evaluation_point_position.append(X_max_damage + 
                                           radius * math.cos(theta))
        Y_evaluation_point_position.append(Y_max_damage + 
                                           radius * math.sin(theta))

    plt.scatter(local_X_position, local_Y_position, c=local_damage, cmap='jet')
    plt.colorbar(orientation="vertical")
    plt.scatter(X_max_damage, Y_max_damage, color = "black", marker='*', s=100,
                label='maximum damage')
    plt.scatter(X_evaluation_point_position, Y_evaluation_point_position, 
                color = 'gray', marker='x', s=50, label='evaluation point')
    plt.axis("equal")
    plt.legend()
    plt.title("Evaluation points in gray for area with label %d"%(label))
    plt.show() 
"""
"""
    # %%===================================================
    '''Evaluate damage on the circle'''
    # =====================================================
    
    #Find the closest points to the evaluation point for polynomial approximation
    approx_damage= []
    sample_size = 13
    for eval_point in np.arange(0,len(X_evaluation_point_position)):        
        X_sample,Y_sample, damage_sample = find_closest_points(
            X_evaluation_point_position[eval_point],
            Y_evaluation_point_position[eval_point],local_X_position,
            local_Y_position, sample_size,local_damage)
        estimated_damage = second_order_polynomial_approximation(X_evaluation_point_position[eval_point],
                                                  Y_evaluation_point_position[eval_point],
                                                  X_sample, Y_sample,
                                                  damage_sample)
        approx_damage.append(estimated_damage) #since centered coordinates
        
        if eval_point == 0 :
            ...#print data
            plt.scatter(local_X_position, local_Y_position, c=local_damage, cmap='jet')
            plt.colorbar(orientation="vertical")
            plt.scatter(X_max_damage, Y_max_damage, color = "black", 
                        marker='*', s=100, label='maximum damage')
            plt.scatter(X_evaluation_point_position[eval_point], 
                        Y_evaluation_point_position[eval_point],
                        color = 'gray', marker='x', s=50, 
                        label='evaluation point')
            ... #plot sample points 
            plt.axis("equal")
            plt.legend()
            plt.title("Sampling points are used to estimate damage at each evaluation point")
            plt.show()   
            
    #To visualize all the points with the same scale :    
    complete_X_position = []
    complete_Y_position = []
    complete_damage = []
    for i in np.arange(0,len(local_X_position)):
        complete_X_position.append(local_X_position[i])
        complete_Y_position.append(local_Y_position[i])
        complete_damage.append(local_damage[i])     
    for i in np.arange(0,len(X_evaluation_point_position)):
        complete_X_position.append(X_evaluation_point_position[i])
        complete_Y_position.append(Y_evaluation_point_position[i])
        complete_damage.append(approx_damage[i])  
        
    plt.scatter(complete_X_position, complete_Y_position, c= complete_damage, cmap='jet')
    plt.colorbar(orientation="vertical")
    plt.scatter(X_max_damage, Y_max_damage, color = "black", marker='*', s=100)
    plt.axis("equal")
    plt.title("Damage estimated at each evaluation point")
    plt.show()   
    
    ...
"""
"""     
    # %%===================================================
    '''Damage evolution analysis to find relative max values'''
    # =====================================================

    wrapped_approx_damage = [approx_damage[-1]]
    wrapped_evaluation_angles = [evaluation_angles[0]-angular_accuracy]
    for i in np.arange(0,len(approx_damage)):
        wrapped_approx_damage.append(approx_damage[i])
        wrapped_evaluation_angles.append(evaluation_angles[i])
    wrapped_approx_damage = np.array(wrapped_approx_damage) 
         
    max_indices = scipy.signal.argrelmax(wrapped_approx_damage)[0]
    max_indices = np.array(max_indices)
    print("At first, damage is max for angles: ")
    for index in max_indices: 
        print(wrapped_evaluation_angles[index] )

    plt.plot(wrapped_evaluation_angles,wrapped_approx_damage,"*-", color = "blue")
    plt.title("Evolution of damage with the angular position between %d"
              %wrapped_evaluation_angles[0]+
              " and %d degrees"%wrapped_evaluation_angles[-1])
    plt.show()
"""
"""  
    # %%===================================================
    '''Damage evolution analysis on smoothed curved 
    to find relative max values'''
    # =====================================================
    
    #Need to be less dependent on the data discretization
    #More radii or smoothing
    window_length = 7
    polyorder = 2
    smoothed_curve = scipy.signal.savgol_filter(approx_damage, window_length, polyorder, mode="wrap")
    wrapped_smoothed_curve = [smoothed_curve[-1]]
    for i in np.arange(0,len(smoothed_curve)):
        wrapped_smoothed_curve.append(smoothed_curve[i])
    wrapped_smoothed_curve = np.array(wrapped_smoothed_curve) 

    max_indices = scipy.signal.argrelmax(wrapped_smoothed_curve)[0]        
    max_indices = np.array(max_indices)
    print("After smoothing, damage is max for angles: ")
    for index in max_indices: 
        print(wrapped_evaluation_angles[index] )
        
    plt.plot(wrapped_evaluation_angles,wrapped_approx_damage,"*-", 
             color = "blue", label = "raw")
    plt.plot(wrapped_evaluation_angles,wrapped_smoothed_curve,"*-", 
             color = "red", label = "smoothed")
    plt.title("Evolution of damage with the angular position between %d"
          %wrapped_evaluation_angles[0]+
          " and %d degrees"%wrapped_evaluation_angles[-1])
    plt.legend()
    plt.show()
"""
"""
    # %%===================================================
    '''Plot segments'''
    # =====================================================

    plt.scatter(local_X_position, local_Y_position, c=local_damage, cmap='jet')
    plt.colorbar(orientation="vertical")
    plt.scatter(X_max_damage, Y_max_damage, color = "black", marker='*', s=100,
                label = "maximum damage")     
    for num_branch in np.arange(0,len(max_indices)):
        X_ridge_point = X_max_damage + radius*math.cos(wrapped_evaluation_angles[max_indices[num_branch]] * math.pi /180. )
        Y_ridge_point = ...
        plt.plot([X_max_damage,X_ridge_point],[Y_max_damage, Y_ridge_point], c="black")
    plt.axis("equal")
    plt.title("Ridge directions for area with label %d"%(label))
    plt.legend()
    plt.show()     
"""
"""
    # %%===================================================
    '''Evaluate damage along the maximum damage direction'''
    # =====================================================
    
    delta_L = 0.2 * radius
    for index_angle in max_indices: 
        print("Damage is max for angle ", wrapped_evaluation_angles[index_angle] ) 
        X_on_segment = []
        Y_on_segment = []
        damage_on_segment =[]
        for multip in np.arange(0,20): 
            new_radius = multip * delta_L 
            X_point = X_max_damage + new_radius*math.cos(wrapped_evaluation_angles[index_angle] * math.pi /180. )
            Y_point = ...
            X_sample,Y_sample, damage_sample = find_closest_points(X_point,
                Y_point,local_X_position, local_Y_position, sample_size,
                local_damage)
            estimated_damage = second_order_polynomial_approximation(...)
            X_on_segment.append(X_point)
            Y_on_segment.append(Y_point)
            damage_on_segment.append(estimated_damage)
                  
        #To visualize all the points with the same scale only:    
        complete_X_position = []
        complete_Y_position = []
        complete_damage = []
        for i in np.arange(0,len(local_X_position)):
            complete_X_position.append(local_X_position[i])
            complete_Y_position.append(local_Y_position[i])
            complete_damage.append(local_damage[i])     
        for i in np.arange(0,len(X_on_segment)):
            complete_X_position.append(X_on_segment[i])
            complete_Y_position.append(Y_on_segment[i])
            complete_damage.append(damage_on_segment[i])  
            
        plt.scatter(complete_X_position, complete_Y_position, c= complete_damage, cmap='jet')
        plt.colorbar(orientation="vertical")
        plt.scatter(X_max_damage, Y_max_damage, color = "black", marker='*', 
                    s=100, label = "maximum damage")
        plt.axis("equal")
        plt.legend()
        plt.title("Evolution of damage for angle: %d deg"%wrapped_evaluation_angles[index_angle] )
        plt.show()  
     
"""
"""
        # %%===================================================
        '''Evaluate crack increment length'''
        # =====================================================
        
        #Select only the points with damage greater than limit_damage
        limit_damage =  0.95
        
        plt.plot(X_on_segment, damage_on_segment, color ='black', 
                 label = "damage")
        plt.plot(X_on_segment, limit_damage*np.ones(len(X_on_segment)), 
                 color ='red', label = "limit value of %f"%limit_damage)
        plt.title("Evolution of damage along the X-axis for angle: %d deg"%wrapped_evaluation_angles[index_angle] )
        plt.legend()
        plt.show() 
        
        selected_X_on_segment = []
        selected_Y_on_segment = []
        for index_point in np.arange(0,len(X_on_segment)):
            estimated_damage = damage_on_segment[index_point]
            if ...:
                selected_X_on_segment.append(X_on_segment[index_point])
                selected_Y_on_segment.append(Y_on_segment[index_point])
        
        plt.scatter(local_X_position, local_Y_position, c=local_damage, cmap='jet')
        plt.colorbar(orientation="vertical")
        plt.scatter(X_max_damage, Y_max_damage, color = "black", marker='*',
                    s=100, label = "maximum damage")
        plt.plot(selected_X_on_segment, selected_Y_on_segment,"-*", c="black")
        plt.axis("equal")   
        plt.legend()
        plt.title("Part of the segments above limit damage for angle: %d deg"%wrapped_evaluation_angles[index_angle] )
        plt.show() 
        
        #Build segment with max damage point and last selected point
        if len(selected_X_on_segment)>0:
            X_all_segments.append([X_max_damage,selected_X_on_segment[-1]])
            Y_all_segments.append([Y_max_damage,selected_Y_on_segment[-1]])
 

# %%===================================================
'''Visualize final result'''
# =====================================================

plt.scatter(X_position, Y_position, c=damage, cmap='jet', marker = ".")
plt.colorbar(orientation="vertical")
for index in np.arange(0,len(X_all_segments)):
    plt.plot(X_all_segments[index], Y_all_segments[index],"-*", c="black")
plt.axis("equal") 
plt.title("Crack increments ready for insertion for time %f"%selected_time)
plt.show()          
"""       
        
