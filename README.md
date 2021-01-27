# Pokemon-Image-Classification

Synopsis:

This is a project to classify images of Pokemon into either water types or grass types. In order to do this I built a convolutional neural network (CNN) to attempt to distinguish the differences between the two types. In our dataset there are 93 Pokémon with grass as a primary type, and 126 Pokémon with water as a primary type. Our dataset comes from Kaggle (https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types). Many water Pokémon resemble aquatic animals (turtles, jellyfish, whales) and are almost always blue or have blue accents. Similarly, grass Pokémon often resemble flora and are almost always green or have green accents. Because of these stark differences we expect the CNN to do relatively well. The place where the CNN might struggle is Pokémon which don’t follow the above conventions. 

Methodology:

The first step was to split both grass and water Pokémon into their own training, test, and validations sets. From there we used data augmentation to create additional data to train on. Data augmentation creates slight variations on existing photos, so there is more data to train on. Due the small sample size initially, only five new photos were generated for each validation set, and three new photos for each training set. Each photo was also scaled down by a factor of 1/255. 
With the data processing of the way, the next step is to build the neural network. For this project we tried three variations, pre-trained model, model built by us with data augmentation, and model without data augmentation.

The only difference between the two networks is one uses data augmentation and one doesn’t.  Each layer has a shape of (3,3), and input shape of (150, 150) and is down sampled by a factor of 2 using max pooling. Max pooling reduces the dimensionality of the layer, avoiding overfitting and helping speed up computation times. A dropout rate of 0.5 is also used. Dropout is also used to prevent overfitting. When a node (and its corresponding edges) are dropped from the model it removes co-dependencies between neurons, reducing overfitting. After 4 layers are added, a dense layer is used to create a fully connected network with 512 nodes. Finally, a dense layer is added, using the sigmoid function to set up a binary classification problem.  The final step is to determine the optimal number of epochs to try. 

Each model was trained on the optimal number of epochs before making comparisons. At first, I tried 20, however based on the graph, 9 looked like a more optimal number of epochs. The optimal number of epochs is where the training and validations diverge the most. However, 9 gave a lower test score than 20. Next we tried 50, which confirmed the optimal number of epochs is 20. The optimal number of epochs for the pretrained model is nine and the optimal number of epochs for the model without data augmentation is 12. 






