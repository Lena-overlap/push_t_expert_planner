"""Physical response records and targeted recovery planning."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from joint_lookahead import Proposal, loss, APP_LOW, APP_HIGH, LOW, HIGH
from geometry import rot, predict_iou, CAPS


@dataclass
class Outcome:
    face: str
    mode: str
    distance: float
    start: np.ndarray
    end: np.ndarray
    travel: float
    elapsed: int
    clear: bool
    phase: str
    eef: np.ndarray

    @property
    def delta(self):
        d=self.end-self.start
        d[2]=np.arctan2(np.sin(d[2]),np.cos(d[2]))
        return d

class Feedback:
    def __init__(self):
        self.records={}
        self.pruned=0
    def add(self,key,outcome): self.records.setdefault(key,[]).append(outcome)
    def for_face(self,key,face): return [x for x in self.records.get(key,[]) if x.face==face and not x.mode.startswith('rotation')]
    def skip(self,key,p):
        if p.mode.startswith('rotation'): return False
        obs=[x for x in self.for_face(key,p.face) if x.mode==p.mode]
        # Two independent commanded distances with effectively identical outcomes.
        # Only prune the same primitive and longer lengths; keep shorter trims
        # and other contacts eligible. No generalization across parent states.
        for i,a in enumerate(obs):
            for b in obs[i+1:]:
                if abs(a.distance-b.distance)<.004: continue
                if p.distance<max(a.distance,b.distance)-.0001:continue
                if a.clear!=b.clear or a.phase!=b.phase:continue
                if a.clear and (max(a.travel,b.travel)>min(a.distance,b.distance)+.003 or abs(a.travel-b.travel)>.001):continue
                if np.linalg.norm(a.end[:2]-b.end[:2])>.0008 or abs(np.arctan2(np.sin(a.end[2]-b.end[2]),np.cos(a.end[2]-b.end[2])))>.012:continue
                if np.linalg.norm(a.eef-b.eef)>.004:continue
                return True
        return False
    def rate(self,key,face,prior):
        samples=[x for x in self.for_face(key,face) if x.clear and x.travel>.004 and np.linalg.norm(x.delta[:2])<.22]
        if not samples:return np.asarray(prior,float)
        # Robust local response in the object frame. Longer reliable trials
        # receive more weight; prior is retained to prevent unstable estimates.
        measured=np.stack([np.r_[rot(-x.start[2])@x.delta[:2]/x.travel,x.delta[2]/x.travel] for x in samples])
        med=np.median(measured,axis=0);med[:2]=np.clip(med[:2],-2.5,2.5);med[2]=np.clip(med[2],-22,22)
        w=min(.85,sum(x.travel for x in samples)/(sum(x.travel for x in samples)+.014))
        return (1-w)*np.asarray(prior)+w*med


def explicit_rotations(node,goal,heading):
    """Enumerate reachable torque-producing contacts instead of blocking just one.
    Estimates are scheduling heuristics, not calibrated predictions.
    """
    from rotation_controller import CONTACTS, local_lever, LOW, HIGH
    q=node.q;err=-float(q[2]);remaining=400-node.steps
    if abs(err)<np.deg2rad(28) or remaining<155:return []
    yaw=heading+q[2];pos=np.asarray(goal)+rot(heading)@q[:2];out=[]
    for c in CONTACTS:
        rate=local_lever(c)/.0035
        if rate*err<=0 or abs(rate)<1:continue
        n=rot(yaw)@c.normal; approach=pos+rot(yaw)@c.xy-.024*n
        if np.any(approach<LOW) or np.any(approach>HIGH):continue
        d=min(.05,abs(err/rate));p=q+np.r_[rot(q[2])@c.normal*d,rate*d];p[2]=np.arctan2(np.sin(p[2]),np.cos(p[2]))
        # Value different angular directions/drift and leave budget for translations.
        cost=.55*float(loss(p))+.10*float(loss(q))+.00008*np.linalg.norm(node.eef[:2]-approach)*1000
        out.append(Proposal('rotation:'+c.name,0.,cost,'rotation_contact',(),min(135,remaining-70),0.))
    return sorted(out,key=lambda p:p.estimate)[:4]


def angle_delta(a, b):
    return float(np.arctan2(np.sin(a-b), np.cos(a-b)))


def endpoint_delta(a, b):
    d=np.asarray(a,float)-np.asarray(b,float)
    d[2]=angle_delta(a[2],b[2]);return d


def terminal_candidates(feedback,key,node,faces,goal,heading,threshold=.85):
    """Select new commanded lengths using a local secant / bias-corrected fit.

    Single samples support only small local changes. Two distinct same-mode
    outcomes support interpolation and tightly bounded extrapolation. Reject
    flat/saturated responses, already tried lengths, unsafe reach, or actions
    unlikely to fit the real remaining time. Predictions do not certify success.
    """
    q=np.asarray(node.q,float); remaining=400-node.steps; options=[]
    if remaining<35:return []
    for face in faces:
        raw=feedback.for_face(key,face)
        for mode in ('push','trim'):
            samples=[x for x in raw if x.mode==mode and x.clear and x.travel>.003 and x.elapsed<=remaining and np.isfinite(x.end).all() and np.isfinite(x.start).all()]
            if not samples:continue
            samples.sort(key=lambda x:x.distance)
            ious=[float(predict_iou(x.end)) for x in samples]; best=max(ious)
            if best<max(.60,threshold-.22):continue
            f=faces[face]; normal=rot(heading+q[2])@f.normal
            point=np.asarray(goal)+rot(heading)@q[:2]+rot(heading+q[2])@f.xy
            approach=point-.024*normal
            if np.any(approach<APP_LOW) or np.any(approach>APP_HIGH):continue
            cap=CAPS.get(face,.075); fits=[]
            # Prefer commanded-distance secants. Do not extrapolate across
            # large differences, inconsistent slopes, or a saturated stroke.
            for a,b in zip(samples,samples[1:]):
                dd=b.distance-a.distance
                if dd<.002 or dd>.030:continue
                delta=endpoint_delta(b.end,a.end); slope=delta/dd
                if np.linalg.norm(delta[:2])<.0007 and abs(delta[2])<.012:continue
                if np.linalg.norm(slope[:2])>3.0 or abs(slope[2])>30:continue
                lo=max(.0025,a.distance-min(.003,.25*dd)); hi=min(cap,b.distance+min(.003,.25*dd))
                fits.append((a,slope,lo,hi,max(a.elapsed,b.elapsed),True))
            # One measured endpoint also provides a local displacement slope,
            # anchored at the command actually issued, not at actual travel.
            # Do not extrapolate from a face which has shown saturation.
            flat=any(abs(a.distance-b.distance)>.003 and np.linalg.norm(endpoint_delta(a.end,b.end)*[1,1,.07])<.0015 for a,b in zip(samples,samples[1:]))
            if not flat:
                a=samples[int(np.argmax(ious))]; slope=a.delta/a.travel
                if np.linalg.norm(slope[:2])<=2.5 and abs(slope[2])<=25:
                    radius=min(.005,max(.002,.18*a.distance))
                    fits.append((a,slope,max(.0025,a.distance-radius),min(cap,a.distance+radius),a.elapsed,False))
            for a,v,lo,hi,elapsed,secant in fits:
                if hi<=lo:continue
                for d in np.linspace(lo,hi,25):
                    if any(abs(d-x.distance)<.0015 for x in raw if x.mode==mode):continue
                    end=point+(d+.018)*normal
                    if np.any(end<LOW) or np.any(end>HIGH):continue
                    time=int(np.ceil(elapsed+max(0,d-a.distance)/.0025+3))
                    if time>remaining:continue
                    pred=a.end+(d-a.distance)*v; pred[2]=angle_delta(pred[2],0.)
                    iou=float(predict_iou(pred))
                    # Only use the scarce extra budget if the physical evidence
                    # predicts substantial improvement and near-task completion.
                    if iou<max(threshold-.025,best+.012):continue
                    cost=.75*float(loss(pred))+.026*(1-iou)+.00004*time-.008*(iou>=.88)
                    if not secant:cost+=.003
                    options.append(Proposal(face,float(d),cost,mode,(),time,iou))
    chosen=[]
    for p in sorted(options,key=lambda p:p.estimate):
        if not any(p.face==o.face and p.mode==o.mode and abs(p.distance-o.distance)<.003 for o in chosen):chosen.append(p)
        if len(chosen)>=4:break
    return chosen


def evidence_roots(roots,archive,feedback,key_fn):
    """Recover parents whose *actual* children nearly succeeded, even if
    their own IoU is low. The normal rescue root selector can miss these."""
    nodes={key_fn(n):n for n in archive+roots}
    existing={key_fn(n):n for n in roots}
    ranked=[]
    for k,n in nodes.items():
        if not n.metrics['upright'] or not n.metrics['on_table'] or n.steps>365:continue
        outcomes=[x for x in feedback.records.get(k,[]) if x.clear and not x.mode.startswith('rotation')]
        if not outcomes:continue
        best=max(float(predict_iou(x.end)) for x in outcomes)
        if best>=.60:ranked.append((best,k,n))
    for _,k,n in sorted(ranked,key=lambda x:(-x[0],x[2].steps))[:6]:existing[k]=n
    return list(existing.values())


def waypoint_proposals(node,goal,heading,feedback=None,key=None):
    """Early, partial angular maneuvers rather than committing to alignment.
    Shortening a long rotation can preserve reach and a subsequent translation.
    Explicitly separate new contact/target combinations from tried full turns.
    """
    yaw=abs(np.rad2deg(node.q[2])); remaining=400-node.steps
    if yaw<80 or remaining<205 or node.steps>175:return []
    from rotation_controller import CONTACTS, local_lever
    contacts={c.name:c for c in CONTACTS}; choices=[]
    for p in explicit_rotations(node,goal,heading):
        name=p.face.split(':',1)[1];c=contacts[name]
        rate=local_lever(c)/.0035
        # Use real response for this exact parent/contact when available.
        # The original full-turn primitive and the partial variant share
        # approach and initial servo; do not retry an observed stalled contact.
        observed=[]
        if feedback is not None:
            for x in feedback.records.get(key,[]):
                if not x.mode.startswith('rotation') or x.face.split(':')[:2]!=['rotation',name] or not x.clear:continue
                gain=abs(np.rad2deg(x.start[2]))-abs(np.rad2deg(x.end[2]))
                if gain>8.:observed.append((gain,x))
            if rotation_stalled(feedback,key,name):continue
        # Move 30–45 degrees rather than aim at near-zero regardless of reach.
        changes=[30.,45.]
        if observed:
            gain,x=max(observed,key=lambda z:z[0]); changes=[max(15.,min(40.,.65*gain)),max(20.,min(55.,.85*gain))]
        for change in changes:
            residual=round(yaw-change,1)
            if residual<25:continue
            d=abs(np.deg2rad(change)/rate)
            end=node.q+np.r_[rot(node.q[2])@c.normal*d,np.sign(rate)*np.deg2rad(change)]
            if observed:
                gain,x=max(observed,key=lambda z:z[0]);end=node.q+endpoint_delta(x.end,x.start)*(change/gain)
            end[2]=angle_delta(end[2],0.)
            cost=.75*float(loss(end))+.00012*min(105,remaining-95)
            choices.append(Proposal(f'rotation:{name}:{residual:.1f}',0.,cost,'rotation_waypoint',(),min(105,remaining-95),float(predict_iou(end))))
    # Distinct contact families first; waypoint choices are still tested in physics.
    choices.sort(key=lambda p:p.estimate)
    diverse=[];seen=set()
    for p in choices:
        name=p.face.split(':')[1]
        if name not in seen:diverse.append(p);seen.add(name)
    for p in choices:
        if p not in diverse:diverse.append(p)
    return diverse[:4]


def rotation_stalled(feedback,key,name):
    """Only exact-parent, explicit-contact failures; no cross-state blocking."""
    for x in feedback.records.get(key,[]):
        if not x.mode.startswith('rotation') or x.face.split(':')[:2]!=['rotation',name]:continue
        d=endpoint_delta(x.end,x.start)
        if x.clear and x.elapsed>=35 and np.linalg.norm(d[:2])<.001 and abs(d[2])<np.deg2rad(1.):return True
    return False


def saturated_trim(feedback,key,p):
    """Adaptive trim can travel farther than requested. Detect convergence to
    the same settled state regardless of command, even in that case. Only
    skip commands bracketed by or slightly beyond two observed lengths.
    """
    if p.mode!='trim':return False
    obs=[x for x in feedback.for_face(key,p.face) if x.clear and x.mode=='trim' and x.travel>.004]
    for i,a in enumerate(obs):
        for b in obs[i+1:]:
            lo,hi=sorted((a.distance,b.distance))
            if hi-lo<.004 or not lo<=p.distance<=hi+.25*(hi-lo):continue
            d=endpoint_delta(a.end,b.end)
            if abs(a.travel-b.travel)>.001 or abs(a.elapsed-b.elapsed)>4 or a.phase!=b.phase:continue
            if np.linalg.norm(d[:2])>.0008 or abs(d[2])>np.deg2rad(.8) or np.linalg.norm(a.eef-b.eef)>.004:continue
            return True
    return False
