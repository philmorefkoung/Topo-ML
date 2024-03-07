import numpy as np
from gtda.homology import CubicalPersistence
from gtda.diagrams import BettiCurve
import pandas as pd

# Load the dataset
data = np.load('pneumoniamnist_128.npz')
train_images = data['train_images']
train_labels = data['train_labels']

# Initialize CubicalPersistence and BettiCurve
CP = CubicalPersistence(
    homology_dimensions=[0, 1],
    coeff=3,
    n_jobs=-1
)
BC = BettiCurve()

# Feature Extraction
data = []
for i, img in enumerate(train_images):
    diagram = CP.fit_transform(img[None, :, :])
    y_betti_curves = BC.fit_transform(diagram)
    # Flatten Betti curves and include image ID and label
    betti_features = np.reshape(y_betti_curves, -1)
    entry = [i + 1, train_labels[i]] + list(betti_features)
    data.append(entry)

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
df.to_csv("pneum_train_betti.csv", index=False)
