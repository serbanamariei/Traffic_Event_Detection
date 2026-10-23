import os
for s in ['train', 'val']:
    im = {f[:-4] for f in os.listdir(f'/content/a9_yolo/images/{s}')}
    lb = {f[:-4] for f in os.listdir(f'/content/a9_yolo/labels/{s}')}
    print(s, len(im), len(lb), 'imagini fara eticheta:', len(im - lb), 'etichete fara imagine:', len(lb - im))