import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder

# ─── CONFIG ───────────────────────────────────────────
ROOT_DIR       = r"D:\al_test\pythonXvs\Automated Traffic Sign Project\automate 3\traffic_Data"
TRAIN_CSV      = os.path.join(ROOT_DIR, "Train.csv")
TEST_CSV       = os.path.join(ROOT_DIR, "Test.csv")
META_CSV       = os.path.join(ROOT_DIR, "Meta.csv")
IMG_SIZE       = (64, 64)
MIN_IMAGES     = 150       # drop classes with fewer than this (after blur filter)
MAX_IMAGES     = 500      # cap classes at this many images (after blur filter)
BLUR_THRESHOLD = 100       # images below this variance are skipped
# ──────────────────────────────────────────────────────

# ─── CLASS NAMES (GTSRB 43 classes) ───────────────────
class_names = {
    0:  'Speed limit 20',       1:  'Speed limit 30',
    2:  'Speed limit 50',       3:  'Speed limit 60',
    4:  'Speed limit 70',       5:  'Speed limit 80',
    6:  'End speed limit 80',   7:  'Speed limit 100',
    8:  'Speed limit 120',      9:  'No passing',
    10: 'No passing >3.5t',     11: 'Right of way',
    12: 'Priority road',        13: 'Yield',
    14: 'Stop',                 15: 'No vehicles',
    16: 'No vehicles >3.5t',    17: 'No entry',
    18: 'General caution',      19: 'Curve left',
    20: 'Curve right',          21: 'Double curve',
    22: 'Bumpy road',           23: 'Slippery road',
    24: 'Road narrows right',   25: 'Road work',
    26: 'Traffic signals',      27: 'Pedestrians',
    28: 'Children crossing',    29: 'Bicycles crossing',
    30: 'Ice/Snow',             31: 'Wild animals',
    32: 'End restrictions',     33: 'Turn right ahead',
    34: 'Turn left ahead',      35: 'Go straight',
    36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right',           39: 'Keep left',
    40: 'Roundabout',           41: 'End no passing',
    42: 'End no passing >3.5t'
}


# ─── HELPER FUNCTIONS ─────────────────────────────────
def fix_brightness(img_array):
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def is_blurry(img_array, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold
# ──────────────────────────────────────────────────────


# ─── 1. LOAD CSVs ─────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
print(f"Train CSV: {len(train_df)} rows")
print(f"Test  CSV: {len(test_df)} rows")


# ─── 2. LOAD ALL TRAIN IMAGES (no filtering yet) ──────
print("\nLoading train images...")
raw_images = {}   # class_name → list of img_arrays
failed     = []
blurry     = 0

for _, row in train_df.iterrows():
    img_path = os.path.join(ROOT_DIR, row['Path'].replace('/', os.sep))
    try:
        img = Image.open(img_path).convert('RGB')
        x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), \
                          int(row['Roi.X2']), int(row['Roi.Y2'])
        img = img.crop((x1, y1, x2, y2))
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)

        # skip blurry images
        if is_blurry(img_array):
            blurry += 1
            continue

        # fix brightness
        img_array = fix_brightness(img_array)

        class_name = class_names.get(int(row['ClassId']), f"Class_{row['ClassId']}")
        if class_name not in raw_images:
            raw_images[class_name] = []
        raw_images[class_name].append(img_array)

    except Exception as e:
        failed.append((img_path, str(e)))

print(f"Blurry:  {blurry} images skipped")
print(f"Failed:  {len(failed)} images")
print(f"Classes loaded: {len(raw_images)}")


# ─── 3. FILTER BY CLASS SIZE (after blur removal) ─────
train_images = []
train_labels = []
dropped      = []

for class_name, imgs in raw_images.items():
    # drop classes with too few images after blur filter
    if len(imgs) < MIN_IMAGES:
        dropped.append((class_name, len(imgs)))
        continue

    # undersample if too many
    if len(imgs) > MAX_IMAGES:
        imgs = imgs[:MAX_IMAGES]

    # add original images
    train_images.extend(imgs)
    train_labels.extend([class_name] * len(imgs))

print(f"\nClasses kept:    {len(raw_images) - len(dropped)}")
print(f"Classes dropped: {len(dropped)} (< {MIN_IMAGES} images after blur filter)")


# ─── 4. LOAD TEST IMAGES ──────────────────────────────
print("\nLoading test images...")
test_images = []
test_labels = []
failed_test = []
blurry_test = 0

for _, row in test_df.iterrows():
    img_path = os.path.join(ROOT_DIR, row['Path'].replace('/', os.sep))
    try:
        img = Image.open(img_path).convert('RGB')
        x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), \
                          int(row['Roi.X2']), int(row['Roi.Y2'])
        img = img.crop((x1, y1, x2, y2))
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)

        if is_blurry(img_array):
            blurry_test += 1
            continue

        img_array = fix_brightness(img_array)
        test_images.append(img_array)
        test_labels.append(class_names.get(int(row['ClassId']), f"Class_{row['ClassId']}"))

    except Exception as e:
        failed_test.append((img_path, str(e)))

print(f"Loaded:  {len(test_images)} test images")
print(f"Blurry:  {blurry_test} images skipped")
print(f"Failed:  {len(failed_test)} images")


# ─── 5. EDA ───────────────────────────────────────────
plt.figure(figsize=(16, 5))
sns.countplot(data=pd.DataFrame({"sign": train_labels}), x="sign",
              order=pd.Series(train_labels).value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title(f"Train Class Distribution — {len(train_images)}")
plt.tight_layout()
plt.show()
 
plt.figure(figsize=(16, 5))
sns.countplot(data=pd.DataFrame({"sign": test_labels}), x="sign",
              order=pd.Series(test_labels).value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title(f"Test Class Distribution — {len(test_images)} images")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("Sample Train Images (Cropped + Brightness Fixed)", fontsize=16)
indices = np.random.choice(len(train_images), 16, replace=False)
for ax, idx in zip(axes.flatten(), indices):
    ax.imshow(train_images[idx])
    ax.set_title(train_labels[idx], fontsize=7)
    ax.axis('off')
plt.tight_layout()
plt.show()

meta_df = pd.read_csv(META_CSV)
fig, axes = plt.subplots(5, 9, figsize=(18, 10))
fig.suptitle("GTSRB — All 43 Classes (Meta)", fontsize=16)
for ax, (_, row) in zip(axes.flatten(), meta_df.iterrows()):
    img_path = os.path.join(ROOT_DIR, row['Path'].replace('/', os.sep))
    class_id = int(row['ClassId'])
    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.set_title(f"{class_id}: {class_names.get(class_id, '')}", fontsize=6)
    except:
        ax.set_title(f"{class_id}: error", fontsize=6)
    ax.axis('off')
for ax in axes.flatten()[len(meta_df):]:
    ax.axis('off')
plt.tight_layout()
plt.show()


# ─── 6. CONVERT TO NUMPY + NORMALIZE ──────────────────
X_train = np.array(train_images) / 255.0
X_test  = np.array(test_images)  / 255.0

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")


# ─── 7. ENCODE LABELS ─────────────────────────────────
le = LabelEncoder()
le.fit(train_labels + test_labels)

y_train = le.transform(train_labels)
y_test  = le.transform(test_labels)

print(f"\ny_train shape: {y_train.shape}")
print(f"y_test  shape: {y_test.shape}")
print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")


# ─── 8. SAVE TO DISK ──────────────────────────────────
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy",  X_test)
np.save("y_test.npy",  y_test)
np.save("classes.npy", le.classes_)

print("\nSaved X_train.npy, y_train.npy, X_test.npy, y_test.npy, classes.npy ✅")