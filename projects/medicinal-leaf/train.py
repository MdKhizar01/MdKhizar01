"""Reconstructed handcrafted image feature classification baseline."""
import argparse
import colorsys
import json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def features(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB').resize((64, 64)), dtype=float) / 255
    hsv = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in rgb.reshape(-1, 3)])
    hist = []
    for data in [rgb.reshape(-1, 3), hsv]:
        for channel in range(3):
            counts, _ = np.histogram(data[:, channel], bins=16, range=(0, 1))
            hist.extend(counts / counts.sum())
    gray = rgb @ np.array([.299, .587, .114])
    quantized = np.minimum((gray * 16).astype(int), 15)
    glcm = np.zeros((16, 16))
    np.add.at(glcm, (quantized[:, :-1].ravel(), quantized[:, 1:].ravel()), 1)
    glcm = (glcm + glcm.T) / (2 * glcm.sum())
    i, j = np.indices(glcm.shape)
    texture = [np.sum(glcm * (i-j)**2), np.sum(glcm**2), np.sum(glcm / (1 + abs(i-j)))]
    center = gray[1:-1, 1:-1]
    codes = np.zeros(center.shape, dtype=np.uint16)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1+dy:63+dy, 1+dx:63+dx]
        codes |= (neighbor >= center).astype(np.uint16) << bit
    lbp = np.bincount(codes.ravel(), minlength=256).astype(float)
    lbp /= lbp.sum()
    # Hu invariant moments of a simple green foreground mask.
    mask = ((rgb[:,:,1] > rgb[:,:,0] * 1.1) & (rgb[:,:,1] > rgb[:,:,2] * 1.1)).astype(float)
    y, x = np.indices(mask.shape)
    mass = mask.sum()
    hu = np.zeros(7)
    if mass:
        x = x - (mask*x).sum()/mass; y = y - (mask*y).sum()/mass
        def eta(p,q): return (mask*x**p*y**q).sum()/mass**(1+(p+q)/2)
        a,b,c,d,e,f,g = eta(2,0),eta(0,2),eta(1,1),eta(3,0),eta(1,2),eta(2,1),eta(0,3)
        u,v,w,z = d-3*e, 3*f-g, d+e, f+g
        hu = np.array([a+b,(a-b)**2+4*c*c,u*u+v*v,w*w+z*z,
            u*w*(w*w-3*z*z)+v*z*(3*w*w-z*z),
            (a-b)*(w*w-z*z)+4*c*w*z,
            v*w*(w*w-3*z*z)-u*z*(3*w*w-z*z)])
        hu = -np.sign(hu)*np.log10(np.maximum(abs(hu),1e-30))
    return np.concatenate([hist, texture, lbp, hu])


def make_demo(folder):
    rng = np.random.default_rng(42)
    for label in range(3):
        target = folder / f'synthetic_{label}'; target.mkdir(parents=True, exist_ok=True)
        for n in range(24):
            image = Image.new('RGB', (96,96), 'white'); draw = ImageDraw.Draw(image)
            shade = int(rng.integers(70,180))
            bounds = (20+label*4, 8, 76-label*4, 88)
            draw.ellipse(bounds, fill=(20,shade,20+label*45))
            draw.line((48,12,48,86), fill=(10,40,10), width=label+1)
            image.save(target / f'{n}.png')


def train(folder, output):
    paths, labels = [], []
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            for path in sorted(sub.iterdir()):
                if path.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}:
                    paths.append(path); labels.append(sub.name)
    if len(set(labels)) < 2:
        raise ValueError('Need at least two class folders containing images')
    x = np.stack([features(p) for p in paths])
    a,b,c,d = train_test_split(x, labels, stratify=labels, test_size=.2, random_state=42)
    model = make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver='full'),
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    model.fit(a,c)
    report = classification_report(d, model.predict(b), output_dict=True, zero_division=0)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output/'model.joblib')
    (output/'metrics.json').write_text(json.dumps({'images':len(paths),'classes':len(set(labels)), 'report':report},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=Path)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output',type=Path,default=Path('artifacts'))
    args=parser.parse_args()
    if bool(args.data) == args.demo:
        parser.error('Choose exactly one of --data or --demo')
    folder = args.data or args.output/'demo_images'
    if args.demo: make_demo(folder)
    print(json.dumps(train(folder,args.output),indent=2))
