import matplotlib.pyplot as plt
import numpy as np

def plot_pred_grid(net, X_test, Y_test, grid_size:int=8, 
                   only_wrong=False):

    Y_test_label = Y_test
    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1)

    for ax in axes.flat:
        # Select random indices
        random_index = np.random.randint(10000)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X_test[random_index].reshape((28,28))
        
        # Display the image

        ax.imshow(X_random_reshaped, cmap='gray')
        y_pred = net.predict(X_random_reshaped.reshape((1, 784)))
        y_pred_label = np.argmax(y_pred, axis=1)
        
        
        # Display the label above the image
        ax.set_title(f'{Y_test_label[random_index]} , {y_pred_label[0]}')
        ax.set_axis_off()