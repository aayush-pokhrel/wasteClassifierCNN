#Waste Classifier CNN ♻️

A binary image classifier that tells recyclable waste apart from general waste.

Credits

This project is built on my professor's CNN template. Thanks to ubibot2000 for the original code and the class: 👉 https://github.com/ubibot2000/Mask_std

The original project classifies masked vs. unmasked faces. I kept the same pipeline and code structure, and trained it on my own waste dataset instead.

Why waste?

Korea has strict waste sorting rules, and putting something in the wrong bin is an easy mistake. The idea is a model that can look at a piece of trash and say which bin it belongs in.

Label	Class	Examples
0	Recyclable Waste	plastic bottles, cans, glass, paper, cardboard
1	General Waste	wrappers, tissues, plastic bags, food packaging
How it works
origin/  ──►  Image Generator  ──►  aug/  ──►  New Fit  ──►  model_trained_data.h5  ──►  Predict new  ──►  "Recyclable (99%)"
(photos)      (augmentation)       (copies)    (training)     (trained model)              (prediction)
Image Generator New(Binary).ipynb: makes 30 augmented copies of every image (random up/down and left/right shifts), so 20 photos per class become 600.
New Fit(Binary).ipynb: loads the augmented images, splits them into train / validation / test, trains a small CNN for 20 epochs, and saves the model.
Predict new.ipynb: loads the saved model and predicts on images it has never seen.
Model
Input (128x128x3)
→ Conv2D(32) → MaxPool
→ Conv2D(64) → MaxPool → Dropout(0.25)
→ Flatten → Dense(128) → Dropout(0.5)
→ Dense(1, sigmoid)      # output near 0 = Recyclable, near 1 = General
Project structure
├── origin/
│   ├── 0/        # 20 recyclable images
│   └── 1/        # 20 general waste images
├── predict/      # 10 held-out images for testing
├── Image Generator New(Binary).ipynb
├── New Fit(Binary).ipynb
├── Predict new.ipynb
└── pyproject.toml

aug/ and model_trained_data.h5 are not uploaded. They get created when you run the notebooks.

Results
	
Training images (after augmentation)	1,160
Split	Train 742 / Val 186 / Test 232
Final validation accuracy	98.9%
Test accuracy	98.28%
Held-out predict/ images	10 / 10 correct


Then run the notebooks in this order:

Image Generator New(Binary).ipynb
New Fit(Binary).ipynb
Predict new.ipynb

If you re-run the generator, delete the aug/ folder first. Otherwise it adds more copies on top of the old ones.

Dataset

The images come from TrashNet by Gary Thung and Mindy Yang (MIT License): https://github.com/garythung/trashnet

Recyclable = glass, paper, cardboard, plastic and metal classes. General = the "trash" class.

Limitations
The test accuracy is optimistic. Augmented copies of the same photo can end up in both the training and test sets, so the model has partly seen the test images before. The predict/ folder is the more honest test.
All TrashNet photos have a plain white background, so the model may struggle with real photos taken on a desk, in a bin, or outdoors.
Some items are tricky: chip bags look like plastic but count as general waste. More data and more classes would help.
Built with

Python · TensorFlow / Keras · OpenCV · scikit-learn · NumPy · Matplotlib
