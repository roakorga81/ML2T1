import numpy as np
import matplotlib.pyplot as plt
from unsupervised_jim import SVD

image = plt.imread(".\images\Ronald.jpeg")
image = np.mean(image, axis=2)

singular_values = np.array([1,5,10,20,50,100,200,500,1000])

# Visualización de la matriz image
plt.subplot(3, 4, 1)
plt.imshow(image)
plt.title('image gray')

for index,sing_val in enumerate(singular_values):
    svd_jim = SVD(sing_val)
    img_reconstructed = svd_jim.fit_transform(image) 
    plt.subplot(3, 4, index+2)
    plt.imshow(img_reconstructed)
    plt.title(f'Singular val: {sing_val}')
    
    # Calculate the RMSE between the original image and the reconstructed image
    rmse = np.sqrt(np.mean((image - img_reconstructed)**2))
    plt.xlabel(f"RMSE {rmse}")

plt.show()