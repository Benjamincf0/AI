import matplotlib.pyplot as plt
import numpy as np

def plot_pred_grid(net, X_test, Y_test, shape:tuple[int, int]=[8, 8], 
                   only_wrong=False):

    fig, axes = plt.subplots(shape[0],shape[1], figsize=(shape[1],shape[0]))
    fig.tight_layout()

    y_pred = np.argmax(net.predict(X_test), axis=1)
    y_test = np.argmax(Y_test, axis=1)

    print(y_pred.shape, y_test.shape)

    if only_wrong:
        poss_indices = np.where(y_pred != y_test)[0]
        print(poss_indices.shape)
    else:
        poss_indices = np.arange(y_test.shape[0])

    random_sample = np.random.choice(poss_indices, size=shape[0]*shape[1], replace=False)

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = random_sample[i]
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X_test[random_index].reshape((28,28))
        
        # Display the image

        ax.imshow(X_random_reshaped, cmap='gray')
        y_pred_val = y_pred[random_index]
        y_test_val = y_test[random_index]
        
        # Display the label above the image
        ax.set_title(f'L:{y_test_val} P:{y_pred_val}')
        ax.set_axis_off()