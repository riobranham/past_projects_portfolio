# ==============================================================================
# Title: Image Processing and Regression
# Author: Rio Branham
# ==============================================================================

# %% Setup

import re
import numpy as np
import vslr as vs
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage import io, color, transform

CHROME_DRIVER_PATH = ('/path/to/chromedriver.exe')
DOWNLOAD_DIR = vs.os.getcwd() + '/data'
LONG_PRICE = 5.5
SHORT_PRICE = 3.25

# %% Read and Format .png Files

# Get Image Files and Names
pngs = vs.search_files('.*\\.png$', top='./data/', recursive=False)
services = [int(re.sub(r'\D*([0-9]+).*', r'\1', png)) for png in pngs]

# Clean Service Number for Joining of Images and Target
targets = pdfs[['quoted_materials', 'service_number']]
targets.service_number = targets.service_number.apply(lambda x:
                                                      int(re.sub('\D', '', x)))
targets = targets.drop('quoted_materials', 1)
# Create Perimeter Target Variable
targets['perim'] = targets.quoted_materials.transform(lambda x: x / 1.54)

# Process Images
init_cad = io.call_plugin('imread', fname=pngs[0])
init_cad = color.rgb2gray(init_cad)
init_cad = transform.resize(init_cad, (612, 792)).ravel().reshape(1, -1)
cads = init_cad
i = 2
for png in pngs[1:]:
    print('Processing image {}/895: {:.2%}'.format(i, i / len(pngs)))
    i += 1
    cad = io.call_plugin('imread', fname=png)
    cad = color.rgb2gray(cad)
    cad = transform.resize(cad, (612, 792)).ravel().reshape(1, -1)
    cads = np.vstack((cads, cad))

# Save Data
# np.save('./data/cads_pixels', cads)

# Merge Data and Target Variable
pixels = vs.pd.DataFrame(cads).assign(service_number=vs.pd.DataFrame(services))
full = pixels.merge(targets)
full = full.drop('service_number', 1)

# Save Aligned Data and Target
target = np.array(full.perim)
data = full.drop('perim', 1)
del full
data = np.array(data)
np.save('./data/cad_data', data)
np.save('./data/cad_target', target)

# %% Models

# Import Data
data = np.load('./data/cad_data.npy')
target = np.load('./data/cad_target.npy')

# Feature Extraction PCA seems to do better than NMF and much faster
pca = PCA(n_components=500)
pca_data = pca.fit_transform(data)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(pca_data, target)

# Gradient Boosting (Best Model So Far)
# Best so far .255 R^2 with 696 samples, 1000 trees, 500 pca_data
# components, random_state=0, learning_rate=.01 and max_depth=5
grbt = GradientBoostingRegressor(learning_rate=.01, max_depth=5,
                                 n_estimators=1000, random_state=0)
grbt.fit(X_train, y_train)
grbt.score(X_test, y_test)
y_pred = grbt.predict(X_test)

# %% Outputs
compare = vs.pd.DataFrame({'actual_perimeters': y_test,
                           'predicted_perimeters': y_pred})
compare['actual_material_cost'] = compare.actual_perimeters.transform(
    lambda x: x * 1.54
)
compare['predicted_material_cost'] = compare.predicted_perimeters.transform(
    lambda x: x * 1.54
)
compare = compare.append(vs.pd.DataFrame({
    'actual_perimeters': compare.actual_perimeters.sum(),
    'predicted_perimeters': compare.predicted_perimeters.sum(),
    'actual_material_cost': compare.actual_material_cost.sum(),
    'predicted_material_cost': compare.predicted_material_cost.sum()},
    index=list(range(len(y_pred), len(y_pred) + 1))))
compare['names'] = compare.index
compare.names = compare.names.apply(str)
compare.names[174] = 'Totals'
compare.to_csv('./data/test_predict_comparison.csv', index=False)
