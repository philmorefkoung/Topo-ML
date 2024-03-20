import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import pandas as pd

# Load the dataset
data = np.load('pneumoniamnist_64.npz')

# Initialize CubicalPersistence and BettiCurve
CP = CubicalPersistence(
    homology_dimensions=[0, 1],
    coeff=3,
    n_jobs=-1
)
BC = BettiCurve()

def process_and_save_data(images, labels, data_type):
    data = []
    for i, img in enumerate(images):
        diagram = CP.fit_transform(img[None, :, :])
        y_betti_curves = BC.fit_transform(diagram)
        # Flatten Betti curves and include image ID and label
        betti_features = np.reshape(y_betti_curves, -1)
        entry = [i + 1, labels[i]] + list(betti_features)
        data.append(entry)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    filename = f"pneum_{data_type}_betti.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

# Process and save train, validation, and test data
process_and_save_data(data['train_images'], data['train_labels'], 'train')
process_and_save_data(data['val_images'], data['val_labels'], 'val')
process_and_save_data(data['test_images'], data['test_labels'], 'test')
