#https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
# Self-Organizing Map, or SOM, falls under the rare domain of unsupervised learning in Neural Networks. 
# Its essentially a grid of neurons, each denoting one cluster learned during training. Traditionally speaking, 
# there is no concept of neuron ‘locations’ in ANNs. However, in an SOM, each neuron has a location, and neurons 
# that lie close to each other represent clusters with similar properties. Each neuron has a weightage vector, 
# which is equal to the centroid of its particular cluster.
#http://www.ai-junkie.com/ann/som/som1.html
#https://genome.tugraz.at/MedicalInformatics2/SOM.pdf

#clustering using tensorflow


#For plotting the images
from matplotlib import pyplot as plt
import numpy as np
from som import SOM
 
#Training inputs for RGBcolors
colors = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']
 
#Train a 20x30 SOM with 400 iterations
som = SOM(20, 30, 3, 400)
som.train(colors)
 
#Get output grid
image_grid = som.get_centroids()
 
#Map colours to their closest neurons
mapped = som.map_vects(colors)
 
#Plot
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
