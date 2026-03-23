"""
Test script: Run current enhancement pipeline on sample images and save results.
"""
import os, sys, torch, torch.nn as nn, numpy as np, cv2
from torchvision import models, transforms
from PIL import Image

# ── Setup model ──
DEVICE = torch.device("cpu")
CLASSES = ['blue_tint','blurry','green_tint','hazy','high_contrast',
           'low_illumination','noisy','raw-890','red_tint']

model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2), nn.Linear(model.last_channel, 512), nn.ReLU(),
    nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, len(CLASSES))
)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Enhancement functions (same as app.py) ──
def enhance_blue_tint(img):
    a = np.array(img).astype(np.float32)/255.0
    a[:,:,0] *= 1.2; a[:,:,2] *= 0.8
    for i in range(3):
        ch = a[:,:,i]; a[:,:,i] = (ch-ch.min())/(ch.max()-ch.min()+1e-6)
    return Image.fromarray(np.clip(a*255,0,255).astype(np.uint8))

def enhance_green_tint(img):
    a = np.array(img).astype(np.float32)/255.0
    a[:,:,1] *= 0.7; a[:,:,0] *= 1.15; a[:,:,2] *= 1.15
    return Image.fromarray(np.clip(a*255,0,255).astype(np.uint8))

def enhance_red_tint(img):
    a = np.array(img).astype(np.float32)/255.0
    a[:,:,0] *= 0.8
    return Image.fromarray(np.clip(a*255,0,255).astype(np.uint8))

def enhance_blurry(img):
    a = np.array(img)
    g = cv2.GaussianBlur(a,(0,0),2.0); s1 = cv2.addWeighted(a,1.8,g,-0.8,0)
    k = np.array([[0,-0.8,0],[-0.8,4.2,-0.8],[0,-0.8,0]])
    s2 = cv2.filter2D(a,-1,k); f = cv2.addWeighted(s1,0.6,s2,0.4,0)
    return Image.fromarray(cv2.convertScaleAbs(f,alpha=1.1,beta=5))

def enhance_hazy(img):
    a = np.array(img).astype(np.float32)/255.0
    for i in range(3):
        ch = a[:,:,i]; a[:,:,i] = (ch-ch.min())/(ch.max()-ch.min()+1e-6)
    return Image.fromarray((np.clip(a*1.2,0,1)*255).astype(np.uint8))

def enhance_low_illumination(img):
    a = np.array(img)
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB); l,aa,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    return Image.fromarray(cv2.cvtColor(cv2.merge([clahe.apply(l),aa,b]),cv2.COLOR_LAB2RGB))

def enhance_noisy(img):
    a = np.array(img)
    d = cv2.fastNlMeansDenoisingColored(a,None,30,30,7,35)
    d = cv2.GaussianBlur(d,(5,5),2.0); d = cv2.GaussianBlur(d,(5,5),2.0)
    return Image.fromarray(cv2.medianBlur(d,7))

def enhance_high_contrast(img):
    a = np.array(img).astype(np.float32)
    a = cv2.addWeighted(a,0.8,a,0,50); a = ((a/255.0)**0.9)*255
    return Image.fromarray(np.clip(a,0,255).astype(np.uint8))

def enhance_raw_890(img):
    a = np.array(img).astype(np.float32)/255.0
    for i in range(3):
        ch = a[:,:,i]; a[:,:,i] = (ch-ch.min())/(ch.max()-ch.min()+1e-6)
    a = np.clip(a*1.15,0,1)
    a = (a*255).astype(np.uint8); a = cv2.GaussianBlur(a,(0,0),1.0)
    return Image.fromarray(cv2.addWeighted(a,1.5,a,-0.5,0))

EMAP = {'blue_tint':enhance_blue_tint,'green_tint':enhance_green_tint,
        'red_tint':enhance_red_tint,'blurry':enhance_blurry,'hazy':enhance_hazy,
        'low_illumination':enhance_low_illumination,'noisy':enhance_noisy,
        'high_contrast':enhance_high_contrast,'raw-890':enhance_raw_890}

# ── Process images ──
test_dir = 'static/uploads'
out_dir = 'static/test_results'
os.makedirs(out_dir, exist_ok=True)

images = ['test_green.jpg', 'test_blue.jpg']
for fname in images:
    path = os.path.join(test_dir, fname)
    if not os.path.exists(path):
        print(f"SKIP: {path} not found"); continue
    
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t); probs = torch.softmax(out,1)
        conf, idx = torch.max(probs,1)
        cls = CLASSES[idx.item()]; cpct = conf.item()*100
    
    print(f"\n{'='*60}")
    print(f"Image: {fname}")
    print(f"Predicted: {cls} (confidence: {cpct:.1f}%)")
    print(f"All probs: ", {CLASSES[i]: round(p*100,1) for i,p in enumerate(probs[0].tolist())})
    
    enhanced = EMAP.get(cls, enhance_raw_890)(img)
    enhanced.save(os.path.join(out_dir, f"enhanced_{fname}"))
    print(f"Saved: enhanced_{fname}")

print("\nDone!")
