"""
Examples:
    python main_traffic.py --folder frames/ --weights best.pt
    python main_traffic.py --folder frames/ --weights best.pt --out results/

Priject Command:
    python YOLOv8/main_traffic.py --folder YOLOv8/tumtraf_yolo/val/video_frames --weights YOLOv8/best.pt

Thresholds and options (tracker, video, accident mode, colors, ...) are in TUNING / COLORS below.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import main_frames as mf

NAN = float('nan')

# Tunable parameters
# Speeds are in km/h, accelerations in m/s^2.
TUNING = dict(
    tracker='bytetrack.yaml',    # or 'botsort.yaml'
    accident_mode='fused',       # 'yolo' = YOLO-only verdict (for comparison)
    auto_wrong_way=False,        # wrong-way detection from the dominant traffic flow (off by default: unreliable on two-way roads)
    conf=0.25,                   # general detection threshold
    acc_conf=0.30,               # YOLO threshold for the accident class (fusion makes the final call)
    imgsz=960, min_frames=3, max_gap=1, margin=0.10,
    stride=1,                    # process every N-th frame
    fps=10.0,                    # used only when file names have no timestamp
    device=None,                 # e.g. 'cpu' or '0'; None = automatic
    # shot (scene cut) detection
    cut_mad=25.0,                # mean abs. difference of 64x36 grey frames above which a cut is declared
    cut_corr=0.5,                # histogram correlation below which (with mad > cut_mad/2) a cut is declared
    gap_secs=1.0,                # a time gap between consecutive frames larger than this is also a cut
    # speed estimation
    win=0.8,                     # window (s) for speed and acceleration
    smooth=5,                    # median filter window on positions (samples)
    min_track=8,                 # shorter tracks are ignored
    fov=70.0,                    # assumed horizontal field of view (deg) for the ground-plane model
    car_height=1.5,              # assumed vehicle height (m), used to estimate camera height
    cal_min_samples=40,          # detections needed to calibrate a shot
    pos_noise=0.7,               # assumed position noise (px) after smoothing
    max_noise=2.5,               # speeds noisier than this (km/h) are shown with '~' and ignored by the rules
    size_m=2.2, fallback_min_size=0.06,   # uncalibrated shots: metres per box size, min box size (fraction of width)
    # rules
    v_move=10.0, v_stop=3.0,
    flow_radius=8.0,             # neighbourhood (box diagonals) for the traffic-flow test
    flow_frac=0.5,               # fraction of neighbours that must move for a stop to count
    wrong_cos=-0.5, wrong_secs=1.5, wrong_disp=3.0,
    brake_decel=4.0, brake_secs=0.3, brake_drop=0.5,
    cong_min=6, v_cong=12.0, cong_secs=3.0,
    contact_ioma=0.15,           # box overlap (intersection / smaller area) counted as contact
    contact_secs=0.3,
    drop_ratio=0.4,              # after contact, speed <= ratio * previous speed counts as a drop
    merge_tol=2.0,               # time tolerance (s) when merging YOLO runs with collisions
    acc_confirm=0.60, acc_possible=0.35,
    # rules from Yolov8_training (speeds in box diagonals per second over 1 s windows)
    stop_rel=0.10, move_rel=0.50, min_stop_s=2.0, early_s=2.0, abrupt_s=1.0, min_cues=2,
    ovl_tol=0.35, ovl_frames=3, ovl_win_s=1.5, cluster_k=2.5, max_box_area=0.5,
    track_conf=0.45,             # tracks with a lower mean confidence (e.g. a palm tree seen as 'car') are ignored by the rules
    video=True,                  # write one annotated mp4 per folder
    save_all=False,              # save every annotated frame as jpg, not only those inside events
)

# BGR colours
COLORS = dict(
    normal=(0, 200, 0),          # green: ordinary vehicle
    event=(235, 90, 20),         # blue: vehicle in a traffic event
    involved=(0, 140, 255),      # orange: vehicle involved in an accident
    accident=(0, 0, 255),        # accident box (change to (235, 90, 20) for blue)
)
EVENT_NAME = {'stopped_vehicle': 'stopped vehicle', 'wrong_way': 'wrong way',
              'hard_braking': 'hard braking', 'congestion': 'congestion'}
ACCIDENT_TYPES = ('accident', 'possible_accident')


# helpers

def frame_times(files, args):
    tss = [mf.parse_ts(f.stem) for f in files]
    if all(t is not None for t in tss):
        return [t - tss[0] for t in tss]
    return [i * args.stride / args.fps for i in range(len(files))]


def time_runs(times, mask, min_dur, max_gap=0.5):
    runs, cur = [], None
    for i, m in enumerate(mask):
        if not m:
            continue
        if cur and times[i] - times[cur[1]] <= max_gap:
            cur[1] = i
        else:
            if cur:
                runs.append(cur)
            cur = [i, i]
    if cur:
        runs.append(cur)
    return [(a, b) for a, b in runs if times[b] - times[a] >= min_dur]


def runmed(a, k):
    h = k // 2
    return np.array([np.median(a[max(0, i - h):i + h + 1]) for i in range(len(a))])


def win_vals(k, t0, t1):
    m = (k['t'] >= t0) & (k['t'] <= t1) & np.isfinite(k['spr'])
    return k['spr'][m]


def small_gray(im):
    import cv2
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)


def is_cut(a, b, args):
    import cv2
    mad = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    if mad > args.cut_mad:
        return True
    ha = cv2.normalize(cv2.calcHist([a], [0], None, [32], [0, 256]), None).flatten()
    hb = cv2.normalize(cv2.calcHist([b], [0], None, [32], [0, 256]), None).flatten()
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < args.cut_corr and mad > args.cut_mad / 2


# ============================================================================ speed model

def calibrate(veh_frame, W, H, args):
    f = 0.5 * W / math.tan(math.radians(args.fov) / 2)
    pts = []
    for vs in veh_frame:
        for _, nm, bx, cf in vs:
            hpx = bx[3] - bx[1]
            if cf < 0.4 or nm not in ('car', 'van') or hpx < 8 or min(bx[0], bx[1]) <= 2 or bx[2] >= W - 2 or bx[3] >= H - 2:
                continue
            pts.append((bx[3], hpx))
    if len(pts) < args.cal_min_samples:
        return None
    y, h = np.array(pts).T
    keep = np.ones(len(y), bool)
    for _ in range(4):
        if keep.sum() < 10:
            return None
        a, b = np.polyfit(y[keep], h[keep], 1)
        res = np.abs(h - (a * y + b))
        keep = res <= max(3.0, 2.5 * float(np.median(res[keep])))
    if keep.sum() < 0.5 * len(y):
        return None
    a, b = np.polyfit(y[keep], h[keep], 1)
    if a <= 0.01:
        return None
    yh = -b / a
    ss_res = float(((h[keep] - (a * y[keep] + b)) ** 2).sum())
    ss_tot = float(((h[keep] - h[keep].mean()) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    hcam = args.car_height / a
    if r2 < 0.6 or np.ptp(y[keep]) < 0.12 * H or yh >= y[keep].min() - 5 or yh < -2 * H or not 3.0 <= hcam <= 40.0:
        return None
    return dict(yh=float(yh), hcam=float(hcam), f=f, cx=W / 2, r2=float(r2), n=int(keep.sum()))


def kinematics(t, x, y, sz, ok, args, cal, W):
    n = len(t)
    sp, sg, vx, vy = (np.full(n, NAN) for _ in range(4))
    lag = np.searchsorted(t, t - args.win, side='right') - 1
    if cal:
        d = y - cal['yh']
        good = ok & (d > 8)
        dd = np.where(good, d, np.nan)
        X = (x - cal['cx']) * cal['hcam'] / dd
        Z = cal['f'] * cal['hcam'] / dd
    for i in range(n):
        k = lag[i]
        if k < 0 or not (ok[i] and ok[k]):
            continue
        dt = t[i] - t[k]
        vx[i], vy[i] = (x[i] - x[k]) / dt, (y[i] - y[k]) / dt
        if cal:
            if not (good[i] and good[k]):
                continue
            sp[i] = math.hypot(X[i] - X[k], Z[i] - Z[k]) / dt * 3.6
            dm = min(d[i], d[k])
            sg[i] = 3.6 * cal['f'] * cal['hcam'] / dm ** 2 * args.pos_noise * math.sqrt(2) / dt
        else:
            s = float(sz[k:i + 1].mean())
            sp[i] = math.hypot(x[i] - x[k], y[i] - y[k]) / s / dt * args.size_m * 3.6
            sg[i] = 3.6 * args.size_m * args.pos_noise * math.sqrt(2) / (s * dt) if s >= args.fallback_min_size * W else 99.0
    rel = np.isfinite(sp) & (sg <= args.max_noise)
    spr = np.where(rel, sp, NAN)
    acc, prev = np.full(n, NAN), np.full(n, NAN)
    for i in range(n):
        k = lag[i]
        if k >= 0 and np.isfinite(spr[k]) and np.isfinite(spr[i]):
            prev[i], acc[i] = spr[k], (spr[i] - spr[k]) / 3.6 / (t[i] - t[k])
    return dict(t=t, x=x, y=y, sz=sz, ok=ok, sp=sp, spr=spr, rel=rel, vx=vx, vy=vy, acc=acc, prev=prev)


def build_tracks(veh_frame, times, W, H, args, cal):
    raw = defaultdict(list)
    for j, vs in enumerate(veh_frame):
        for tid, nm, bx, cf in vs:
            raw[tid].append((j, times[j], bx, nm, cf))
    m = max(4.0, 0.01 * W)
    kin = {}
    for tid, s in raw.items():
        if len(s) < args.min_track:
            continue
        j = np.array([a[0] for a in s])
        t = np.array([a[1] for a in s], float)
        bx = np.array([a[2] for a in s], float)
        x, y = (bx[:, 0] + bx[:, 2]) / 2, bx[:, 3]
        sz = np.sqrt(np.maximum((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]), 1.0))
        ok = (bx[:, 0] > m) & (bx[:, 1] > m) & (bx[:, 2] < W - m) & (bx[:, 3] < H - m)
        k = kinematics(t, runmed(x, args.smooth), runmed(y, args.smooth), runmed(sz, args.smooth), ok, args, cal, W)
        k['j'] = j
        k['cls'] = Counter(a[3] for a in s).most_common(1)[0][0]
        k['conf'] = float(np.mean([a[4] for a in s]))
        kin[tid] = k
    return kin


def global_direction(kin, v_move):
    us = []
    for k in kin.values():
        m = k['spr'] >= v_move
        if m.any():
            nrm = np.hypot(k['vx'][m], k['vy'][m])
            us.append(np.stack([k['vx'][m] / nrm, k['vy'][m] / nrm], 1))
    if not us:
        return None
    u = np.concatenate(us)
    if len(u) < 30:
        return None
    mean = u.mean(0)
    nrm = np.linalg.norm(mean)
    return mean / nrm if nrm >= 0.7 else None


# notebook rules (Yolov8_training)
# 2D port of the accident rules from the training notebook: a vehicle that moved and then stayed still
# (diag-normalised speed over 1 s windows), clustering of such vehicles, and the cues abrupt stop /
# several vehicles stopped early / traffic stopped behind / overlap at stop
def frame_period(times):
    d = np.diff(times)
    return float(np.median(d)) if len(d) else 0.1


def rule_params(dt, args):
    K = max(1, round(1.0 / dt))
    return dict(dt=dt, K=K, K2=max(1, round(0.5 / dt)), need=max(1, round(args.min_stop_s / dt) - K + 1),
                w=max(1, round(args.ovl_win_s / dt)))


def rule_tracks(veh_frame, P, args):
    by = {}
    for j, vs in enumerate(veh_frame):
        for tid, nm, bx, cf in vs:
            tr = by.setdefault(tid, dict(oid=tid, xy={}, dg={}, box={}, cf=[], cls=Counter()))
            if j in tr['xy']:
                continue
            tr['xy'][j] = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
            tr['dg'][j] = math.hypot(bx[2] - bx[0], bx[3] - bx[1])
            tr['box'][j] = tuple(bx)
            tr['cf'].append(cf)
            tr['cls'][nm] += 1
    out = []
    for tr in by.values():
        if len(tr['xy']) < args.min_track or float(np.mean(tr['cf'])) < args.track_conf:
            continue
        tr['cls'] = tr['cls'].most_common(1)[0][0]
        K, dt = P['K'], P['dt']
        tr['sp'] = {t: math.dist(tr['xy'][t], tr['xy'][t - K]) / max((tr['dg'][t] + tr['dg'][t - K]) / 2, 1.0) / (K * dt)
                    for t in sorted(tr['xy']) if t - K in tr['xy']}
        out.append(tr)
    return out


def stop_runs(tr, P, args):
    runs, cur, moved = [], [], False
    for t in sorted(tr['sp']):
        v = tr['sp'][t]
        if v >= args.move_rel:
            if len(cur) >= P['need']:
                runs.append((cur[0], cur[-1]))
            cur, moved = [], True
        elif moved and v < args.stop_rel:
            cur.append(t)
        else:
            if len(cur) >= P['need']:
                runs.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= P['need']:
        runs.append((cur[0], cur[-1]))
    first = min(tr['xy'])
    return [(max(first, a - P['K']), b) for a, b in runs]


def stationary_info(tr, P, args):
    sp = tr['sp']
    ts = sorted(sp)
    run = []
    for t in reversed(ts):
        if sp[t] < args.stop_rel:
            run.append(t)
        else:
            break
    if len(run) < P['need']:
        return None
    run_start = run[-1]
    moved = any(sp[t] >= args.move_rel for t in ts if t < run_start)
    first = min(tr['xy'])
    return (max(first, run_start - P['K']) if moved else first), moved, run_start


def last_move(tr, P, args, s):
    last, K2 = None, P['K2']
    for t in sorted(tr['xy']):
        if t - K2 in tr['xy']:
            d = math.dist(tr['xy'][t], tr['xy'][t - K2]) / max((tr['dg'][t] + tr['dg'][t - K2]) / 2, 1.0) / (K2 * P['dt'])
            if d >= args.move_rel and t < s + 2 * K2:
                last = t
    return last


def stop_overlap(c, partners, P, args):
    if not c['moved']:
        return None
    tr, run, first = c['tr'], 0, None
    for t in range(c['s'] - P['w'], c['s'] + P['w'] + 1):
        b, hit = tr['box'].get(t), False
        if b is not None:
            for d in partners:
                db = d['tr']['box'].get(t)
                if d is not c and d['s'] <= t and db is not None and box_overlap(b, db) >= args.ovl_tol:
                    hit = True
                    break
        if hit:
            if run == 0:
                first = t
            run += 1
            if run >= args.ovl_frames:
                return first
        else:
            run, first = 0, None
    return None


def stopped_clusters(tracks, P, args):
    cands = []
    for tr in tracks:
        info = stationary_info(tr, P, args)
        if info is None or not info[1]:
            continue
        s = info[0]
        ts = [t for t in sorted(tr['xy']) if t >= s]
        if not ts:
            continue
        lm = last_move(tr, P, args, s)
        abrupt = lm is not None and (s - lm) * P['dt'] <= args.abrupt_s
        cands.append(dict(tr=tr, s=s, moved=True, abrupt=abrupt, imp=min(s, lm) if abrupt else s,
                          ctr=(float(np.median([tr['xy'][t][0] for t in ts])), float(np.median([tr['xy'][t][1] for t in ts]))),
                          diag=float(np.median([tr['dg'][t] for t in ts]))))
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if math.dist(cands[i]['ctr'], cands[j]['ctr']) < args.cluster_k * (cands[i]['diag'] + cands[j]['diag']) / 2:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, c in enumerate(cands):
        groups[find(i)].append(c)
    clusters = []
    for members in groups.values():
        for c in members:
            c['ovl'] = stop_overlap(c, members, P, args)
            if c['ovl'] is not None:
                c['imp'] = min(c['imp'], c['ovl'])
        trig = [c for c in members if c['abrupt'] or c['ovl'] is not None]
        t_on = min(c['imp'] for c in (trig or members))
        lim = t_on + round(args.early_s / P['dt'])
        early = [c for c in members if c['s'] <= lim]
        late = [c for c in members if c['s'] > lim]
        cues = dict(abrupt_stop=any(c['abrupt'] for c in early), several_stopped_early=len(early) >= 2,
                    traffic_stopped_behind=len(late) >= 1, overlap_at_stop=any(c['ovl'] is not None for c in early))
        clusters.append(dict(members=members, early=early, cues=cues, ok=sum(cues.values()) >= args.min_cues, onset=t_on,
                             ids={c['tr']['oid'] for c in early}, link={c['tr']['oid'] for c in members},
                             j1=max(max(c['tr']['xy']) for c in early)))
    return clusters


# traffic events

def ev(kind, seq, ids, f0, f1, t0, t1, info='', score=None):
    return dict(sequence=seq, type=kind, track_ids=sorted(int(i) for i in ids), start_frame=int(f0), end_frame=int(f1),
                start_s=round(float(t0), 2), end_s=round(float(t1), 2), info=info,
                score=None if score is None else round(float(score), 3))


def traffic_events(name, times, kin, rtracks, P, args):
    n = len(times)
    kin = {tid: k for tid, k in kin.items() if k['conf'] >= args.track_conf}
    state = defaultdict(list)      # frame -> [(id, reliable speed km/h, acceleration m/s2)]
    spr_at = {}
    for tid, k in kin.items():
        spr_at[tid] = {int(j): s for j, s in zip(k['j'], k['spr'])}
        for i in range(len(k['t'])):
            state[int(k['j'][i])].append((tid, k['spr'][i], k['acc'][i]))
    events = []

    # congestion: many vehicles with a reliable, low speed; only the slow ones are reported
    cmask = []
    for j in range(n):
        sps = [s[1] for s in state.get(j, []) if np.isfinite(s[1])]
        cmask.append(len(sps) >= args.cong_min and float(np.median(sps)) < args.v_cong)
    cong = []
    for a, b in time_runs(times, cmask, args.cong_secs, max_gap=1.0):
        ids = set()
        for tid, m in spr_at.items():
            v = [m[j] for j in range(a, b + 1) if j in m and np.isfinite(m[j])]
            if len(v) >= 3 and float(np.median(v)) < args.v_cong:
                ids.add(tid)
        if len(ids) >= args.cong_min:
            cong.append(ev('congestion', name, ids, a, b, times[a], times[b], f'{len(ids)} vehicles'))
    events += cong
    in_cong = lambda j0, j1: any(e['start_frame'] <= j1 and j0 <= e['end_frame'] for e in cong)

    def moving(tid, j, tr=None):
        v = spr_at.get(tid, {}).get(j, NAN)
        r = (tr or {}).get('sp', {}).get(j, NAN)
        if (np.isfinite(v) and v >= args.v_move) or (np.isfinite(r) and r >= args.move_rel):
            return True
        return False if (np.isfinite(v) or np.isfinite(r)) else None

    # stopped vehicle (notebook rule): it moved, then stayed still >= min_stop_s, while the traffic around it keeps flowing
    # (otherwise it is a red light or a queue)
    for tr in rtracks:
        for a, b in stop_runs(tr, P, args):
            cx = float(np.mean([tr['xy'][t][0] for t in range(a, b + 1) if t in tr['xy']]))
            cy = float(np.mean([tr['xy'][t][1] for t in range(a, b + 1) if t in tr['xy']]))
            if in_cong(a, b):
                continue
            R = args.flow_radius * float(np.median(list(tr['dg'].values())))
            mv = ms = 0
            for o in rtracks:
                if o is tr:
                    continue
                for t in range(a, b + 1):
                    if t in o['xy'] and math.dist(o['xy'][t], (cx, cy)) <= R:
                        m = moving(o['oid'], t, o)
                        if m is not None:
                            ms += 1
                            mv += m
            dur = times[b] - times[a]
            if ms == 0:
                if dur < 2 * args.min_stop_s:
                    continue
            elif mv < 3 or mv / ms < args.flow_frac:
                continue
            events.append(ev('stopped_vehicle', name, [tr['oid']], a, b, times[a], times[b], tr['cls']))

    glob = global_direction(kin, args.v_move) if args.auto_wrong_way else None
    with np.errstate(invalid='ignore'):
        for tid, k in kin.items():
            t, j, sp = k['t'], k['j'], k['spr']

            # wrong way: moving against a strong dominant flow (auto_wrong_way)
            if glob is not None:
                wm, cosv = np.zeros(len(t), bool), np.full(len(t), NAN)
                for i in range(len(t)):
                    if not sp[i] >= args.v_move:
                        continue
                    cosv[i] = (k['vx'][i] * glob[0] + k['vy'][i] * glob[1]) / math.hypot(k['vx'][i], k['vy'][i])
                    wm[i] = cosv[i] < args.wrong_cos
                for a, b in time_runs(t, wm, args.wrong_secs):
                    disp = math.hypot(k['x'][b] - k['x'][a], k['y'][b] - k['y'][a]) / float(np.median(k['sz'][a:b + 1]))
                    if np.nanmean(cosv[a:b + 1]) < args.wrong_cos and disp >= args.wrong_disp:
                        events.append(ev('wrong_way', name, [tid], j[a], j[b], t[a], t[b], k['cls']))

            # hard braking: strong deceleration with a real speed drop, ignoring traffic-wide slowdowns
            bm = (k['acc'] < -args.brake_decel) & (k['prev'] >= args.v_move)
            for a, b in time_runs(t, bm, args.brake_secs):
                peak = np.nanmax(k['prev'][a:b + 1])
                if not sp[b] <= (1 - args.brake_drop) * peak or in_cong(j[a], j[b]):
                    continue
                others = [s for s in state.get(int(j[b]), []) if s[0] != tid and np.isfinite(s[2])]
                if len(others) >= 3 and sum(s[2] < -0.5 * args.brake_decel for s in others) / len(others) >= 0.6:
                    continue
                events.append(ev('hard_braking', name, [tid], j[a], j[b], t[a], t[b],
                                 f'decel {-np.nanmin(k["acc"][a:b + 1]):.1f} m/s2'))
    return events, cong


# collisions

def box_overlap(ba, bb):
    iw, ih = min(ba[2], bb[2]) - max(ba[0], bb[0]), min(ba[3], bb[3]) - max(ba[1], bb[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    area = lambda b: max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / min(area(ba), area(bb))


def collisions(veh_frame, times, kin, args):
    contact = defaultdict(list)
    for j, vs in enumerate(veh_frame):
        for p in range(len(vs)):
            for q in range(p + 1, len(vs)):
                (ta, _, ba, _), (tb, _, bb, _) = vs[p], vs[q]
                if ta == tb or ta not in kin or tb not in kin:
                    continue
                if {kin[ta]['cls'], kin[tb]['cls']} == {'truck', 'trailer'}:
                    continue
                if box_overlap(ba, bb) >= args.contact_ioma:
                    contact[(min(ta, tb), max(ta, tb))].append(j)
    out = []
    for (a, b), js in contact.items():
        runs, cur = [], [js[0], js[0]]
        for j in js[1:]:
            if j - cur[1] <= 2:
                cur[1] = j
            else:
                runs.append(cur)
                cur = [j, j]
        runs.append(cur)
        for j0, j1 in runs:
            if j1 - j0 < 1 or times[j1] - times[j0] < args.contact_secs:
                continue
            tc = times[j0]
            pre, post = {}, {}
            for i in (a, b):
                w = win_vals(kin[i], tc - 1.5, tc - 0.1)
                pre[i] = float(w.max()) if len(w) else NAN
                w = win_vals(kin[i], tc + 0.3, tc + 2.0)
                post[i] = float(np.median(w)) if len(w) else NAN
            if not (np.isfinite(pre[a]) and np.isfinite(pre[b])):
                continue    # one track starts at the contact itself or has no reliable speed
            drops = [i for i in (a, b) if pre[i] >= args.v_move and np.isfinite(post[i])
                     and (post[i] < args.v_stop or post[i] <= args.drop_ratio * pre[i])]
            if not drops:
                continue
            strong = len(drops) == 2 or all(np.isfinite(post[i]) and post[i] < args.v_stop for i in (a, b))
            out.append(dict(t0=tc, t1=times[j1], j0=j0, j1=j1, ids={a, b}, strong=strong))
    return out


# accident fusion

CUE_WEIGHT = dict(overlap_at_stop=0.30, abrupt_stop=0.10, several_stopped_early=0.10, traffic_stopped_behind=0.10)


def fuse_accidents(name, times, kin, rtracks, clusters, veh_frame, accs_frame, flags, best, cong, W, H, args):
    if args.accident_mode == 'yolo':
        evs, rows = [], []
        for a, b, cnt in mf.find_runs(flags, args.max_gap):
            ok = cnt >= max(1, min(args.min_frames, len(times)))
            peak = max(best[a:b + 1])
            rows.append(dict(sequence=name, t0=times[a], t1=times[b], src='yolo', score=peak,
                             status='confirmed' if ok else 'rejected', detail=f'n={cnt} peak={peak:.2f}'))
            if ok:
                evs.append(ev('accident', name, [], a, b, times[a], times[b], f'yolo-only peak {peak:.2f}', peak))
        return evs, rows

    items = []      # candidate evidence from the three sources
    for a, b, cnt in mf.find_runs(flags, args.max_gap):
        ids = set()
        for jj in range(a, b + 1):
            if accs_frame[jj]:
                bx = accs_frame[jj][0][1]
                for tid, _, bb, _ in veh_frame[jj]:
                    if bx[0] <= (bb[0] + bb[2]) / 2 <= bx[2] and bx[1] <= (bb[1] + bb[3]) / 2 <= bx[3]:
                        ids.add(tid)
        items.append(dict(kind='yolo', j0=a, j1=b, ids=ids, link=ids, n=cnt, peak=max(best[a:b + 1])))
    for c in collisions(veh_frame, times, kin, args):
        items.append(dict(kind='contact', j0=c['j0'], j1=c['j1'], ids=set(c['ids']), link=set(c['ids']), coll=c))
    for cl in clusters:
        items.append(dict(kind='rules', j0=int(cl['onset']), j1=int(cl['j1']), ids=set(cl['ids']), link=set(cl['link']), cl=cl))

    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if (times[a['j0']] <= times[b['j1']] + args.merge_tol and times[b['j0']] <= times[a['j1']] + args.merge_tol
                    and (a['link'] & b['link'])):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[find(i)].append(it)

    evs, rows = [], []
    with np.errstate(invalid='ignore'):
        for grp in groups.values():
            yolo = [g for g in grp if g['kind'] == 'yolo']
            coll = [g['coll'] for g in grp if g['kind'] == 'contact']
            rul = [g['cl'] for g in grp if g['kind'] == 'rules']
            n = sum(g['n'] for g in yolo)
            peak = max([g['peak'] for g in yolo], default=0.0)
            cl = max(rul, key=lambda c: sum(CUE_WEIGHT[k] for k, v in c['cues'].items() if v), default=None)
            cc = max(coll, key=lambda c: c['strong'], default=None)
            if cl is not None:
                ids = set(cl['ids'])
            elif cc is not None:
                ids = set(cc['ids'])
            else:
                ids = set().union(*[g['ids'] for g in yolo])
            j0 = min(g['j0'] for g in grp)
            j1 = max(g['j1'] for g in grp)
            if cl is not None:
                j0 = min(j0, int(cl['onset']))
            t0 = times[j0]
            kids = [i for i in ids if i in kin]
            pre, post = {}, {}
            for i in kids:
                w = win_vals(kin[i], t0 - 3.0, t0 + 0.3)
                pre[i] = float(w.max()) if len(w) else NAN
                w = win_vals(kin[i], t0 + 0.3, t0 + 2.5)
                post[i] = float(np.median(w)) if len(w) else NAN
            drop = any(np.isfinite(pre[i]) and np.isfinite(post[i]) and pre[i] >= args.v_move
                       and (post[i] < args.v_stop or post[i] <= args.drop_ratio * pre[i]) for i in kids)
            halted = sum(1 for i in kids if np.isfinite(post[i]) and post[i] < args.v_stop) >= 2
            parts = {}
            if n:
                parts['yolo'] = 0.5 * min(1.0, n / (2.0 * max(1, args.min_frames))) * min(1.0, peak / 0.6)
            if cc is not None:
                parts['contact'] = 0.25 + (0.05 if cc['strong'] else 0.0)
            if cl is not None:
                for k, v in cl['cues'].items():
                    if v:
                        parts[k] = CUE_WEIGHT[k]
            if drop:
                parts['speed_drop'] = 0.15
            if halted:
                parts['halted'] = 0.10
            if cl is not None and not cl['cues']['overlap_at_stop']:
                t_ref = min(j1, int(cl['onset']) + round(args.early_s / frame_period(times)))
                st = [tr['sp'].get(t_ref) for tr in rtracks if tr['sp'].get(t_ref) is not None]
                if len(st) >= 4 and sum(v < args.stop_rel for v in st) / len(st) >= 0.6:
                    parts['global_stop'] = -0.20       # almost everybody is stopped: red light / jam, not a crash
            if not rul and not coll and any(g['j0'] <= e['end_frame'] and e['start_frame'] <= g['j1'] for g in yolo for e in cong):
                parts['in_congestion'] = -0.15
            if not rul and not coll and len(ids) < 2:
                parts['lt2_vehicles'] = -0.10
            score = max(0.0, min(1.0, sum(parts.values())))
            status = 'confirmed' if score >= args.acc_confirm else 'possible' if score >= args.acc_possible else 'rejected'
            src = '+'.join(k for k, has in (('yolo', yolo), ('contact', coll), ('rules', rul)) if has)
            detail = ' '.join(f'{k}={v:+.2f}' for k, v in parts.items())
            rows.append(dict(sequence=name, t0=t0, t1=times[j1], src=src, score=score, status=status,
                             detail=f'n={n} peak={peak:.2f} ids={sorted(ids)} rules_ok={cl["ok"] if cl else None} {detail}'))
            if status != 'rejected':
                kind = 'accident' if status == 'confirmed' else 'possible_accident'
                evs.append(ev(kind, name, ids, j0, j1, t0, times[j1], f'score {score:.2f} [{src}] {detail}', score))
    return evs, rows


def analyze_sequence(name, times, W, H, veh_frame, accs_frame, flags, best, args):
    cal = calibrate(veh_frame, W, H, args)
    kin = build_tracks(veh_frame, times, W, H, args, cal)
    P = rule_params(frame_period(times), args)
    rtracks = rule_tracks(veh_frame, P, args)
    clusters = stopped_clusters(rtracks, P, args)
    events, cong = traffic_events(name, times, kin, rtracks, P, args)
    acc_events, rows = fuse_accidents(name, times, {t: k for t, k in kin.items() if k['conf'] >= args.track_conf}, rtracks, clusters, veh_frame, accs_frame, flags, best, cong, W, H, args)
    for e in events:
        for a in acc_events:
            if a['type'] == 'accident' and set(e['track_ids']) & set(a['track_ids']) and a['start_s'] - 3 <= e['start_s'] <= a['end_s'] + 10:
                e['info'] += ' (part of accident)'
    events += acc_events
    events.sort(key=lambda e: (e['start_s'], e['type']))
    return events, rows, kin, cal


# rendering

def put_label(im, text, x, y, color, fs, th):
    import cv2
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def render_shot(files, veh_frame, accs_frame, kin, events, args, out, name, writer):
    """Box colours: green = vehicle, blue = vehicle in a traffic event, orange = vehicle involved in an accident,
    red = whole accident box. Label above each box: class, confidence, speed [, event name]."""
    import cv2
    spd = {(tid, int(j)): (s, r) for tid, k in kin.items() for j, s, r in zip(k['j'], k['sp'], k['rel'])}
    ev_by, acc_by = defaultdict(lambda: defaultdict(set)), defaultdict(list)
    for e in events:
        for j in range(e['start_frame'], e['end_frame'] + 1):
            if e['type'] in ACCIDENT_TYPES:
                acc_by[j].append(e)
            else:
                for tid in e['track_ids']:
                    ev_by[j][tid].add(EVENT_NAME[e['type']])
    for j, f in enumerate(files):
        active = j in acc_by or j in ev_by
        if not (active or args.save_all or args.video):
            continue
        im = mf.load_image(f)
        if im is None:
            continue
        fs, th = max(0.4, im.shape[1] / 1600), max(1, round(im.shape[1] / 640))
        involved = {tid for e in acc_by.get(j, []) for tid in e['track_ids']}
        for tid, nm, bx, cf in veh_frame[j]:
            names = sorted(ev_by[j][tid]) if j in ev_by and tid in ev_by[j] else []
            col = COLORS['involved'] if tid in involved else COLORS['event'] if names else COLORS['normal']
            x1, y1, x2, y2 = map(int, bx)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 2 * th)
            sp, rel = spd.get((tid, j), (NAN, False))
            speed = f'{"" if rel else "~"}{sp:.0f} km/h' if sp == sp else ''
            wbox = x2 - x1
            parts = [f'{nm.capitalize()} {cf:.2f}', speed] if wbox >= 60 else [f'{cf:.2f}', speed] if wbox >= 28 else []
            label = ' | '.join(p for p in parts + names if p)
            if label:
                put_label(im, label, x1, max(y1 - 5, 12), col, fs, th)
        for e in acc_by.get(j, []):
            if accs_frame[j]:
                conf, bx = accs_frame[j][0]
                tail = f'conf={conf:.2f}'
            else:
                boxes = [b for tid, _, b, _ in veh_frame[j] if tid in e['track_ids']]
                if not boxes:
                    continue
                bx = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
                tail = f'score={e["score"]:.2f}'
            x1, y1, x2, y2 = map(int, bx)
            if (x2 - x1) * (y2 - y1) > args.max_box_area * im.shape[0] * im.shape[1]:
                continue
            cv2.rectangle(im, (x1, y1), (x2, y2), COLORS['accident'], 3 * th)
            put_label(im, f'{"Accident" if e["type"] == "accident" else "Possible accident"} {tail}',
                      x1, max(y1 - 22 * th, 12), COLORS['accident'], fs, th)
        if active or args.save_all:
            (out / name).mkdir(parents=True, exist_ok=True)
            cv2.imencode('.jpg', im)[1].tofile(str(out / name / (f.stem + '.jpg')))
        if args.video:
            if writer['w'] is None:
                writer['w'] = cv2.VideoWriter(str(out / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                              args.fps / max(1, args.stride), (im.shape[1], im.shape[0]))
            writer['w'].write(im)


# pipeline

def extract(res, args):
    """One tracked frame -> (vehicles [(id, class, box, conf)], accident boxes [(conf, box)], flag, best conf)."""
    veh = []
    ids = res.boxes.id
    if ids is not None:
        for b, tid in zip(res.boxes, ids.int().tolist()):
            nm = res.names[int(b.cls[0])].lower()
            cf = float(b.conf[0])
            if nm in mf.NOT_VEHICLE or cf < args.conf:
                continue
            veh.append((tid, nm, b.xyxy[0].tolist(), cf))
    _, accs = mf.analyze(res, args.conf, args.acc_conf, args.margin)
    if accs:
        return veh, [(c, big) for c, big, _ in accs], True, accs[0][0]
    return veh, [], False, 0.0


def process_folder(name, files, times, track_fn, args, out, sink):
    writer = {'w': None}
    state = {'shot': None, 'no': 0}

    def finish(s):
        n = len(s['files'])
        events, rows, kin, cal = analyze_sequence(name, s['times'], s['W'], s['H'], s['veh'], s['accs'], s['flags'], s['best'], args)
        model = (f"calibrated, camera height {cal['hcam']:.1f} m" if cal else 'approximate (uncalibrated)')
        print(f"\n  Shot {s['no']}: {n} frames, t={s['times'][0]:.1f}-{s['times'][-1]:.1f}s, speed model: {model}", flush=True)
        for e in events:
            e['shot'] = s['no']
            e['start_file'], e['end_file'] = s['files'][e['start_frame']].name, s['files'][e['end_frame']].name
            print(f"    {e['type']:18s} +{e['start_s']:.1f}s ... +{e['end_s']:.1f}s | ids {e['track_ids']} | {e['info']}")
        verdict = ('ACCIDENT' if any(e['type'] == 'accident' for e in events) else
                   'POSSIBLE ACCIDENT' if any(e['type'] == 'possible_accident' for e in events) else 'no accident')
        print(f'    => {verdict}', flush=True)
        sink['events'] += events
        sink['rows'] += [dict(r, shot=s['no']) for r in rows]
        sink['summary'].append((name, s['no'], s['times'][0], s['times'][-1], n, len(kin), verdict,
                                sorted({e['type'] for e in events} - set(ACCIDENT_TYPES))))
        for tid, k in kin.items():
            for i in range(len(k['t'])):
                sink['tracks'].append([name, s['no'], tid, k['cls'], int(k['j'][i]), round(float(k['t'][i]), 3), round(float(k['x'][i]), 1),
                                       round(float(k['y'][i]), 1), round(float(k['sp'][i]), 1), int(k['rel'][i]), round(float(k['acc'][i]), 2)])
        render_shot(s['files'], s['veh'], s['accs'], kin, events, args, out, name, writer)

    prev_small, prev_t = None, None
    for j, f in enumerate(files):
        im = mf.load_image(f)
        if im is None:
            continue
        sm = small_gray(im)
        cut = prev_small is not None and (is_cut(prev_small, sm, args) or times[j] - prev_t > args.gap_secs)
        prev_small, prev_t = sm, times[j]
        if state['shot'] is None or cut:
            if state['shot'] is not None:
                finish(state['shot'])
            state['no'] += 1
            state['shot'] = dict(no=state['no'], files=[], times=[], veh=[], accs=[], flags=[], best=[],
                                 W=im.shape[1], H=im.shape[0], first=True)
        s = state['shot']
        res = track_fn(im, persist=not s['first'])      # persist=False resets the tracker at the start of a shot
        s['first'] = False
        veh, accs, flag, best = extract(res, args)
        s['files'].append(f)
        s['times'].append(times[j])
        s['veh'].append(veh)
        s['accs'].append(accs)
        s['flags'].append(flag)
        s['best'].append(best)
    if state['shot'] is not None:
        finish(state['shot'])
    if writer['w'] is not None:
        writer['w'].release()


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True, help='folder with frames; every sub-folder is one sequence')
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--out', default='output_traffic')
    return argparse.Namespace(**{**TUNING, **vars(ap.parse_args(argv))})


def write_outputs(out, sink):
    cols = ['sequence', 'shot', 'type', 'track_ids', 'start_s', 'end_s', 'score', 'start_frame', 'end_frame', 'start_file', 'end_file', 'info']
    with open(out / 'events.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in sink['events']:
            w.writerow([' '.join(map(str, e[c])) if c == 'track_ids' else e.get(c, '') for c in cols])
    (out / 'events.json').write_text(json.dumps(sink['events'], indent=2, ensure_ascii=False), encoding='utf-8')
    with open(out / 'accident_candidates.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['sequence', 'shot', 'start_s', 'end_s', 'source', 'score', 'status', 'detail'])
        for r in sink['rows']:
            w.writerow([r['sequence'], r['shot'], round(r['t0'], 2), round(r['t1'], 2), r['src'], round(r['score'], 3), r['status'], r['detail']])
    with open(out / 'tracks.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['sequence', 'shot', 'track_id', 'class', 'frame', 'time_s', 'x', 'y', 'speed_kmh', 'speed_reliable', 'accel_ms2'])
        w.writerows(sink['tracks'])


def main():
    args = get_args()
    import torch
    from ultralytics import YOLO

    model = YOLO(args.weights)
    prec = mf.precision_kwargs(args.device)
    pred_conf = min(args.conf, args.acc_conf)

    def track_fn(im, persist):
        with torch.no_grad():
            return model.track(im, persist=persist, tracker=args.tracker, conf=pred_conf, imgsz=args.imgsz,
                               device=args.device, verbose=False, **prec)[0]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sink = dict(events=[], rows=[], tracks=[], summary=[])
    for name, files in mf.collect_sequences(args.folder).items():
        files = files[::max(1, args.stride)]
        print(f'\nSequence: {name} ({len(files)} frames)', flush=True)
        process_folder(name, files, frame_times(files, args), track_fn, args, out, sink)
    write_outputs(out, sink)

    print('\n=== SUMMARY ===')
    for name, no, t0, t1, n, nt, verdict, others in sink['summary']:
        print(f'{name[:30]:30s} shot {no:3d} {t0:7.1f}-{t1:7.1f}s {n:5d} frames {nt:4d} tracks  {verdict:18s} {", ".join(others)}')
    print(f'\n{len(sink["events"])} events saved to {out.resolve() / "events.csv"}; accident candidates in accident_candidates.csv')


if __name__ == '__main__':
    main()