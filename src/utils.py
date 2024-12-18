import numpy as np

def combine_main_and_subarrays(train_data):
    """
    Combines the main values (first column) and the subarrays (second column) into a single dataset.
    
    Args:
        train_data (numpy.ndarray): Input data with two columns, where the first column contains the main values
                                     and the second column contains nested arrays.
    
    Returns:
        numpy.ndarray: Combined dataset with main values and subarrays stacked horizontally.
    """
    # Extract the main values (the first column)
    main_values = train_data[:, 0].reshape(-1, 1)  # Reshape to (n_samples, 1) to make it a column

    # Extract the subarrays (the second column)
    subarrays = np.array([arr for arr in train_data[:, 1]])  # Convert nested arrays into a 2D array

    # Stack the main values and subarrays horizontally
    combined_data = np.hstack([main_values, subarrays])

    return combined_data
