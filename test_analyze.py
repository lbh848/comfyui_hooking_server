import requests
import os
import sys

base = r"C:\Users\lbh\Downloads"
# Find the directory with Korean chars
for d in os.listdir(base):
    full = os.path.join(base, d)
    if os.path.isdir(full) and "1 (1)" in d:
        img_dir = full
        break
else:
    print("Directory not found")
    sys.exit(1)

images = [f for f in os.listdir(img_dir) if f.endswith(('.webp', '.png', '.jpg', '.jpeg'))]
if not images:
    print("No images found")
    sys.exit(1)

img_path = os.path.join(img_dir, images[0])
print(f"Testing with: {images[0]}")

with open(img_path, 'rb') as f:
    files = {'image': (images[0], f, 'image/webp')}
    data = {'category': 'expressions'}
    try:
        resp = requests.post('http://127.0.0.1:8189/api/asset_tool/analyze', files=files, data=data, timeout=120)
        print(f"Status code: {resp.status_code}")
        result = resp.json()
        print(f"Success: {result.get('success')}")
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        if result.get('tags'):
            print(f"Tags ({len(result['tags'])}): {result['tags'][:20]}")
    except requests.exceptions.Timeout:
        print("Request timed out after 120s")
    except Exception as e:
        print(f"Request failed: {e}")