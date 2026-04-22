"""2장 quick 검증"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=float)
def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=float)
def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
def euler_xyz(rx, ry, rz):
    return rot_x(rx) @ rot_y(ry) @ rot_z(rz)

class Joint:
    def __init__(self, name, pos=(0,0,0), parent=None):
        self.name = name; self.pos = np.array(pos, dtype=float)
        self.rot = np.zeros(3); self.parent = parent
    def world_transform(self):
        T = np.eye(4); T[:3,:3] = euler_xyz(*self.rot); T[:3,3] = self.pos
        if not self.parent: return T
        return self.parent.world_transform() @ T
    def world_pos(self): return self.world_transform()[:3,3]

def build():
    root = Joint('root', (0, 0.02, 0))
    torso = Joint('torso', (0, 0.055, 0), root)
    neck = Joint('neck', (0, 0.30, 0), torso)
    head = Joint('head', (0, 0.06, 0), neck)
    ls = Joint('l_shoulder', (-0.18, 0.28, 0), torso)
    le = Joint('l_elbow', (0, -0.24, 0), ls)
    lw = Joint('l_wrist', (0, -0.21, 0), le)
    rs = Joint('r_shoulder', (0.18, 0.28, 0), torso)
    re = Joint('r_elbow', (0, -0.24, 0), rs)
    rw = Joint('r_wrist', (0, -0.21, 0), re)
    lh = Joint('l_hip', (-0.085, -0.055, 0), root)
    lk = Joint('l_knee', (0, -0.38, 0), lh)
    la = Joint('l_ankle', (0, -0.36, 0), lk)
    rh = Joint('r_hip', (0.085, -0.055, 0), root)
    rk = Joint('r_knee', (0, -0.38, 0), rh)
    ra = Joint('r_ankle', (0, -0.36, 0), rk)
    all_j = [root,torso,neck,head,ls,le,lw,rs,re,rw,lh,lk,la,rh,rk,ra]
    return {j.name: j for j in all_j}

def parse_body(bf):
    kps = []
    for i in range(0, len(bf), 3): kps.append({'x':bf[i],'y':bf[i+1],'c':bf[i+2]})
    ok = lambda i: i < len(kps) and kps[i]['c'] > 0
    return kps, ok

def apply_limb(joints, kps, ok, side, pIdx, jIdx, cIdx, fix_elbow=True):
    is_arm = pIdx <= 7
    j1 = f'{side}_shoulder' if is_arm else f'{side}_hip'
    j2 = f'{side}_elbow' if is_arm else f'{side}_knee'
    if ok(pIdx) and ok(jIdx):
        vx = kps[jIdx]['x'] - kps[pIdx]['x']
        vy = kps[jIdx]['y'] - kps[pIdx]['y']
        joints[j1].rot[2] = math.atan2(-vy, vx) + math.pi/2
    if ok(pIdx) and ok(jIdx) and ok(cIdx):
        v1x=kps[jIdx]['x']-kps[pIdx]['x']; v1y=kps[jIdx]['y']-kps[pIdx]['y']
        v2x=kps[cIdx]['x']-kps[jIdx]['x']; v2y=kps[cIdx]['y']-kps[jIdx]['y']
        dot=v1x*v2x+v1y*v2y; cross=v1x*v2y-v1y*v2x
        bend = max(0, min(math.pi, math.atan2(abs(cross), dot)))
        joints[j2].rot[0] = -bend if (fix_elbow and is_arm) else bend

def apply_pose(joints, bf, swap=True, fix_elbow=True):
    kps, ok = parse_body(bf)
    for j in joints.values(): j.rot = np.zeros(3)
    if ok(2) and ok(5) and ok(8) and ok(11):
        mxS=(kps[2]['x']+kps[5]['x'])/2; myS=(kps[2]['y']+kps[5]['y'])/2
        mxH=(kps[8]['x']+kps[11]['x'])/2; myH=(kps[8]['y']+kps[11]['y'])/2
        joints['torso'].rot[2] = math.atan2(myH-myS, mxS-mxH) - math.pi/2
    if ok(0) and ok(1):
        vx=kps[0]['x']-kps[1]['x']; vy=kps[0]['y']-kps[1]['y']
        joints['head'].rot[2] = (math.atan2(-vy, vx) - math.pi/2) * 0.7
    if swap:
        apply_limb(joints,kps,ok,'l',2,3,4,fix_elbow)
        apply_limb(joints,kps,ok,'r',5,6,7,fix_elbow)
        apply_limb(joints,kps,ok,'l',8,9,10,fix_elbow)
        apply_limb(joints,kps,ok,'r',11,12,13,fix_elbow)
    else:
        apply_limb(joints,kps,ok,'r',2,3,4,fix_elbow)
        apply_limb(joints,kps,ok,'l',5,6,7,fix_elbow)
        apply_limb(joints,kps,ok,'r',8,9,10,fix_elbow)
        apply_limb(joints,kps,ok,'l',11,12,13,fix_elbow)

def export_pose(joints, swap=True):
    pos = {n: j.world_pos() for n, j in joints.items()}
    vals = list(pos.values())
    mnX=min(p[0] for p in vals); mxX=max(p[0] for p in vals)
    mnY=min(p[1] for p in vals); mxY=max(p[1] for p in vals)
    pad=0.2; pw=(mxX-mnX+pad*2) or 1; ph=(mxY-mnY+pad*2) or 1
    cw=512; ch=round(512*ph/pw)
    def proj(p): return (round(((p[0]-mnX+pad)/pw)*cw*10)/10,
                         round(((mxY+pad-p[1])/ph)*ch*10)/10)
    if swap:
        op_map = ['head','neck','l_shoulder','l_elbow','l_wrist',
                   'r_shoulder','r_elbow','r_wrist','l_hip','l_knee','l_ankle',
                   'r_hip','r_knee','r_ankle','head','head','head','head']
        eye_off = [(-0.025,0.04,14),(0.025,0.04,15),(-0.07,0,16),(0.07,0,17)]
    else:
        op_map = ['head','neck','r_shoulder','r_elbow','r_wrist',
                   'l_shoulder','l_elbow','l_wrist','r_hip','r_knee','r_ankle',
                   'l_hip','l_knee','l_ankle','head','head','head','head']
        eye_off = [(0.025,0.04,14),(-0.025,0.04,15),(0.07,0,16),(-0.07,0,17)]
    body = []
    for i in range(18):
        p = pos.get(op_map[i])
        if p is not None: x,y = proj(p); body.extend([x,y,1.0])
        else: body.extend([0,0,0])
    hp = pos.get('head')
    if hp is not None:
        for dx,dy,idx in eye_off:
            c=proj(np.array([hp[0]+dx,hp[1]+dy,hp[2]]))
            body[idx*3]=c[0]; body[idx*3+1]=c[1]; body[idx*3+2]=1
        np2=proj(np.array([hp[0],hp[1]+0.02,hp[2]+0.09]))
        body[0]=np2[0]; body[1]=np2[1]
    return body, cw, ch

OP = ['Nose','Neck','RShldr','RElbow','RWrist','LShldr','LElbow','LWrist',
      'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','REye','LEye','REar','LEar']

if __name__ == '__main__':
    from PIL import Image
    from modes.dwpose_standalone import DWPoseDetector
    print("Loading detector...")
    det = DWPoseDetector()

    imgs = [
        r'C:\Users\lbh\Downloads\6accf5de-682c-44f2-8eff-cbf352a9defe.jpeg',
        r'C:\Users\lbh\Downloads\02c0b202-abf9-443e-b5a2-53f0708d8b40.png',
    ]

    for path in imgs:
        img = Image.open(path).convert('RGB')
        W, H = img.size
        img_np = np.array(img)
        print(f"\n{'#'*72}")
        print(f"  {os.path.basename(path)}  {W}x{H}")
        print(f"{'#'*72}")

        pose = det.detect(img_np)
        if not pose:
            print("  DETECTION FAILED"); continue
        orig = pose['people'][0]['pose_keypoints_2d']
        cw_o, ch_o = pose['canvas_width'], pose['canvas_height']
        print(f"  Canvas: {cw_o}x{ch_o}")

        # 원본 키포인트
        kps, _ = parse_body(orig)
        for i, k in enumerate(kps):
            if k['c'] > 0:
                print(f"    {i:2d} {OP[i]:<8} ({k['x']:7.1f}, {k['y']:7.1f})")

        # 새 코드 (swap + fix)
        j = build()
        apply_pose(j, orig, swap=True, fix_elbow=True)
        print("\n  Rotations (deg):")
        for n in ['torso','head','l_shoulder','l_elbow','r_shoulder','r_elbow',
                   'l_hip','l_knee','r_hip','r_knee']:
            if any(abs(r) > 0.001 for r in j[n].rot):
                print(f"    {n:14s}: rx={math.degrees(j[n].rot[0]):7.1f} ry={math.degrees(j[n].rot[1]):7.1f} rz={math.degrees(j[n].rot[2]):7.1f}")

        print("\n  World positions:")
        for n in ['head','neck','l_shoulder','l_elbow','l_wrist',
                   'r_shoulder','r_elbow','r_wrist','l_hip','l_knee','l_ankle',
                   'r_hip','r_knee','r_ankle']:
            p = j[n].world_pos()
            print(f"    {n:14s}: ({p[0]:+7.3f}, {p[1]:+7.3f}, {p[2]:+7.3f})")

        rt, cw_r, ch_r = export_pose(j, swap=True)

        # Normalized 비교
        print(f"\n  Normalized comparison (canvas {cw_r}x{ch_r}):")
        print(f"    {'Joint':<8} {'Orig':>18} {'RT':>18} {'Dist':>7}")
        total = 0; cnt = 0
        for i in range(14):
            ox,oy,oc = orig[i*3],orig[i*3+1],orig[i*3+2]
            rx,ry = rt[i*3],rt[i*3+1]
            if oc <= 0: continue
            onx,ony = ox/cw_o, oy/ch_o
            rnx,rny = rx/cw_r, ry/ch_r
            d = math.sqrt((onx-rnx)**2 + (ony-rny)**2)
            total += d; cnt += 1
            flag = " <<<" if d > 0.15 else ""
            print(f"    {OP[i]:<8} ({onx:.3f},{ony:.3f})  ({rnx:.3f},{rny:.3f})  {d:.3f}{flag}")
        if cnt:
            print(f"    Avg: {total/cnt:.3f}")
