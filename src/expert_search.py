"""Hybrid expert search and batch collection entry point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import heapq
import itertools
import numpy as np
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from geometry import rot, CAPS
from contact_planner import CalibratedPlanner, load_model
from dataset import cases, band as original_band, make_expert, DemoRecorder
from joint_lookahead import Lattice, Proposal, augmented, loss, terminal_distance
from sim_checkpoint import Checkpoints, CacheMismatch
from physical_feedback import Feedback, Outcome, explicit_rotations, terminal_candidates, evidence_roots, waypoint_proposals, saturated_trim, rotation_stalled


LIMIT=400
WIDE_BINS=tuple((lo,lo+15) for lo in range(90,180,15))

def band(angle):
    a=abs(float(angle))
    if a<=90: return original_band(angle)
    return next(f'{lo}-{hi}' for lo,hi in WIDE_BINS if lo<=a<hi or (hi==180 and a==180))

def wide_cases(seed,per_bin,angle_seed):
    """Balanced signed errors, strictly inside each 90–180 degree stratum."""
    if per_bin<2 or per_bin%2: raise ValueError('per-bin must be positive and even')
    rng=np.random.default_rng(angle_seed); result=[]
    for lo,hi in WIDE_BINS:
        signs=np.tile([1.,-1.],per_bin//2); rng.shuffle(signs)
        for sign in signs:
            result.append((seed+len(result),float(round(sign*rng.uniform(lo+.10,hi-.10),4))))
    return result
def typical_video_seeds(selected, count):
    """Preselect evenly spaced difficulty strata, alternating yaw signs if available."""
    ordered=sorted(selected,key=lambda item:(abs(item[1]),item[0]))
    if count is None or count>=len(ordered): return {seed for seed,_ in ordered}
    if count<1: raise ValueError('video count must be positive')
    chosen=set()
    for i,indices in enumerate(np.array_split(np.arange(len(ordered)),count)):
        group=[ordered[int(k)] for k in indices]
        target=(abs(group[0][1])+abs(group[-1][1]))/2
        sign=1 if i%2==0 else -1
        candidates=[item for item in group if item[1]*sign>0] or group
        chosen.add(min(candidates,key=lambda item:(abs(abs(item[1])-target),item[0]))[0])
    return chosen

LOW=np.array([-.29,-.28]); HIGH=np.array([.25,.28])
# These are proposal locations, not assertions about physical response.
EXTRA={'xp_upper':('xp_center',-3.0),'xp_lower':('xp_center',3.0),
       'xn_upper':('xn_center',3.0),'xn_lower':('xn_center',-3.0)}

def wrapped(a): return float(np.arctan2(np.sin(a),np.cos(a)))
def valid(m): return bool(m['upright'] and m['on_table'])
def ok(m,env): return valid(m) and m['iou']>=env.overlap_threshold

def pose(env):
    d=env.sim.data
    xy=d.body_xpos[env.cube_body_id,:2].copy()
    R=d.body_xmat[env.cube_body_id].reshape(3,3)
    yaw=math.atan2(R[1,0],R[0,0]); goal=d.body_xpos[env.target_body_id,:2].copy()
    return xy,yaw,goal,np.r_[rot(-env.target_yaw)@(xy-goal),wrapped(yaw-env.target_yaw)]

def signature(env,obs):
    d=env.sim.data
    # Includes robot, T and clutter positions/velocities and the target.
    return np.r_[d.qpos,d.qvel,d.body_xpos[env.target_body_id],
                 d.body_xmat[env.target_body_id],obs['robot0_eef_pos']].copy()

@dataclass
class Node:
    actions: list
    tags: list
    sig: np.ndarray
    q: np.ndarray
    eef: np.ndarray
    metrics: dict
    streak: int=0
    closed: bool=False
    history: tuple=()
    rates: dict=field(default_factory=dict)
    trace: list=field(default_factory=list)
    @property
    def steps(self): return len(self.actions)

class ReplayMismatch(RuntimeError): pass

class Runner:
    def __init__(self,env,reset,stable):
        self.env=env; self.reset=reset; self.stable=stable
        self.executed=0; self.cache=None; self.rescue=False
        self.search_archive=[]; self.search_tried=set(); self.failed_approaches=set(); self.feedback=Feedback()
    def start(self):
        self.obs=self.reset(); self.actions=[]; self.tags=[]; self.trace=[]; self.streak=0
        self.m=self.env.task_metrics()
        self.initial_sig=signature(self.env,self.obs)
        return self.node(False)
    def node(self,closed,history=(),rates=None):
        return Node(self.actions.copy(),self.tags.copy(),signature(self.env,self.obs),pose(self.env)[3],
                    np.asarray(self.obs['robot0_eef_pos']).copy(),dict(self.m),self.streak,closed,history,dict(rates or {}),self.trace.copy())
    def step(self,a,tag):
        if len(self.actions)>=LIMIT: return False
        a=np.asarray(a,float)
        if a.shape!=(7,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:
            raise ValueError('Invalid 7D OSC action')
        self.obs,_,_,_=self.env.step(a.copy()); self.executed+=1
        self.actions.append(a.copy()); self.tags.append(tag)
        self.m=self.env.task_metrics(); self.streak=self.streak+1 if ok(self.m,self.env) else 0
        xy,yaw,_,q=pose(self.env); eef=self.obs['robot0_eef_pos']
        self.trace.append(dict(step=len(self.actions),stage=tag[0],phase=tag[1],iou=self.m['iou'],success=ok(self.m,self.env),success_streak=self.streak,object_x=xy[0],object_y=xy[1],object_yaw_rad=yaw,error_x_m=q[0],error_y_m=q[1],error_yaw_deg=np.rad2deg(q[2]),eef_x=eef[0],eef_y=eef[1],eef_z=eef[2],**{f'action_{i}':float(x) for i,x in enumerate(a)}))
        return True
    def reached(self):
        return self.streak>=self.stable and self.obs['robot0_eef_pos'][2]>=.895
    def replay(self,node,record=False,frame=None):
        init=self.start()
        if not np.allclose(init.sig,self.expected_initial,atol=2e-6,rtol=1e-6):
            raise ReplayMismatch('Reset did not reproduce initial robot/object/clutter state')
        rec=DemoRecorder() if record else None
        if rec: rec.observe(self.obs,self.env,self.m,ok(self.m,self.env))
        if frame is not None: frame(0,self.m,("initial","reset"),ok(self.m,self.env))
        for a,tag in zip(node.actions,node.tags):
            if rec: rec.action(a,*tag)
            self.step(a,tag)
            if rec: rec.observe(self.obs,self.env,self.m,ok(self.m,self.env))
            if frame is not None: frame(len(self.actions),self.m,tag,ok(self.m,self.env))
        if not np.allclose(signature(self.env,self.obs),node.sig,atol=2e-5,rtol=1e-5) or self.streak!=node.streak:
            raise ReplayMismatch('Action replay did not reproduce checkpoint state / success streak')
        return rec
    def move(self,desired,wrist=None,cap=.70,grip=1.):
        eef=np.asarray(self.obs['robot0_eef_pos']); r=np.zeros(3)
        if wrist is not None:
            current=self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id['right']].reshape(3,3)
            r=np.clip(Rotation.from_matrix(wrist@current.T).as_rotvec()/.5,-.5,.5)
        return np.r_[np.clip((np.asarray(desired)-eef)*18.,-cap,cap),r,grip]
    def release(self):
        # Same vertical release as staged baseline; stop only after stable original task success.
        while len(self.actions)<LIMIT and ok(self.m,self.env) and not self.reached():
            eef=np.asarray(self.obs['robot0_eef_pos'])
            self.step(self.move(np.r_[eef[:2],.95]),('search','release'))
        return self.reached()
    def prepare(self,closed):
        # No lateral action until verified lifted. Gripper changes above the T.
        for _ in range(30):
            eef=np.asarray(self.obs['robot0_eef_pos'])
            if eef[2]>=.896: break
            if not self.step(self.move(np.r_[eef[:2],.91],grip=1. if closed else -1.),('search','lift')): return False
        if self.obs['robot0_eef_pos'][2]<.896: return False
        if not closed:
            for _ in range(24):
                if not self.step(np.r_[np.zeros(6),1.],('search','close')): return False
        return valid(self.m) and self.obs['robot0_eef_pos'][2]>=.885


def proposals(node,env,priors,faces,limit):
    rates={**priors,**node.rates}
    lattice=Lattice(rates,faces)
    options=lattice.plan(node.q,node.eef,LIMIT-node.steps,np.asarray(env.target_xy),env.target_yaw,limit,closed=node.closed)
    # Independent rotation remains a fallback, not a required initial stage.
    if LIMIT-node.steps>=100 and abs(node.q[2])>np.deg2rad(18) and limit>=5:
        options.insert(min(4,len(options)),Proposal('rotation',0.,float(loss(node.q)*.6+.012),'rotation'))
    return options[:limit]

def finish_contact(r):
    # Try the normal stable success path first. If success was transient while
    # lifting, keep the physically safe state available for the next correction.
    if r.release(): return True
    return valid(r.m) and r.prepare(True)

def push(r,p,faces):
    from geometry import tee_robot_contact
    from translation_controller import stop_buffer
    from rotation_controller import RepositioningPushTExpert
    env=r.env; f=faces[p.face]; xy,yaw,_,_=pose(env); n=rot(yaw)@f.normal
    point=xy+rot(yaw)@f.xy; approach=point-.024*n
    if np.any(approach<[-.26,-.25]) or np.any(approach>[.21,.25]): return False
    wrist=RepositioningPushTExpert.wrist_rotation(SimpleNamespace(environment=env),n)
    # Approach and descend with original gains, tolerances, and bounded phase duration.
    for name,z,timeout,tol in [('approach',.90,45,.010),('descend',.824,30,.007)]:
        goal=np.r_[approach,z]; done=False
        for _ in range(timeout):
            eef=np.asarray(r.obs['robot0_eef_pos']); a=r.move(goal,wrist)
            if np.linalg.norm(eef-goal)<tol and (name!='approach' or np.linalg.norm(a[3:6])<.25):
                done=True; break
            if name=='approach' and eef[2]<.885: return False
            if not valid(r.m) or not r.step(a,('search',name)): return False
            if ok(r.m,env): return finish_contact(r)
        if not done: return False
    origin=xy.copy(); eef0=np.asarray(r.obs['robot0_eef_pos'])[:2].copy(); last=0.; stalls=0; seek=0; acquired=False; anchor=None
    target_distance=p.distance; samples=[]; last_adapt=-1.; max_distance=min(CAPS.get(p.face,.075),p.distance+.015)
    # Frozen stroke direction for measured distance; material tracking only after verified contact.
    for _ in range(84):
        cur,ang,_,_=pose(env); eef=np.asarray(r.obs['robot0_eef_pos']); motion=float((cur-origin)@n)
        prog=max(0.,motion-last); last=motion
        if ok(r.m,env): return finish_contact(r)
        if not valid(r.m): return False
        # Terminal strokes estimate the local SE(2) response from *actual* motion,
        # and optimize the stopping distance against polygon IoU. Do not change
        # contact or command direction merely because the prior was inaccurate.
        q=pose(env)[3]; samples.append((motion,q.copy()))
        if p.mode=='trim' and motion>=.003 and motion-last_adapt>=.002 and len(samples)>=3 and np.linalg.norm(q[:2])<.035 and abs(q[2])<.65:
            prev=next((x for x in reversed(samples[:-1]) if motion-x[0]>=.003),None)
            if prev is not None:
                d=motion-prev[0]; v=(q-prev[1])/d;v[2]=wrapped(q[2]-prev[1][2])/d
                if np.linalg.norm(v[:2])<3 and abs(v[2])<30:
                    upper=min(.018,max(0.,max_distance-motion),max(0.,(LIMIT-len(r.actions)-12)*.002))
                    planned=motion+terminal_distance(q,v,upper)
                    target_distance=float(np.clip(planned,max(motion,target_distance-.008),max(motion,min(max_distance,target_distance+.008))))
                    last_adapt=motion
        gap=target_distance-motion; _,hit=tee_robot_contact(env)
        acquired |= bool(hit or motion>.0006)
        if hit and anchor is None: anchor=rot(-ang)@(eef[:2]-cur)
        if acquired: stalls=0 if prog>.00025 else stalls+1
        else: seek+=1
        if gap<=stop_buffer(prog,margin_m=.00065,lookahead_steps=1.2,maximum_m=.0035) or seek>=24 or stalls>=10: break
        if (eef[:2]-eef0)@n>target_distance+.035: break
        if abs(wrapped(ang-yaw))>np.deg2rad(38): break
        cap=float(np.clip(gap*20,.055,.70)) if acquired else .24
        cmd=n*cap; desired_wrist=wrist
        # Co-rotate with material face if turning. Do not chase slipped contact beyond envelope.
        if anchor is not None and abs(wrapped(ang-yaw))>np.deg2rad(4):
            live=rot(ang)@f.normal; t=np.array([-live[1],live[0]]); err=cur+rot(ang)@anchor-eef[:2]
            if live@n<.79 or abs(err@t)>.018 or abs(err@live)>.025: break
            cmd=live*np.clip(min(cap,.34)+14*(err@live),.035,.40)+t*np.clip(24*(err@t),-.20,.20)
            rz=np.eye(3); rz[:2,:2]=rot(wrapped(ang-yaw)); desired_wrist=rz@wrist
        if cmd@n<=0 or np.any(eef[:2]+.05*cmd<LOW) or np.any(eef[:2]+.05*cmd>HIGH): break
        a=r.move(np.r_[eef[:2],.824],desired_wrist,cap=.25); a[:2]=cmd
        if not r.step(a,('joint','trim' if p.mode=='trim' else 'push')): return False
    if ok(r.m,env): return finish_contact(r)
    # Disengage along the last face normal, then rise; require real clearance.
    eef=np.asarray(r.obs['robot0_eef_pos']); target=np.r_[np.clip(eef[:2]-.016*n,LOW,HIGH),eef[2]]
    for _ in range(18):
        if np.linalg.norm(np.asarray(r.obs['robot0_eef_pos'])-target)<.006: break
        if not valid(r.m) or not r.step(r.move(target),('search','retreat')): return False
        if ok(r.m,env): return finish_contact(r)
    return r.prepare(True)

def rotation_trial(r, mode="rotation", contact=None):
    from rotation_controller import ContinuousRotation as ClosedTipRotation
    remaining=LIMIT-len(r.actions)
    extended=mode in ('rotation_multi','rotation_alt')
    budget=min(175,max(85,remaining-85)) if extended else min(105,remaining)
    obj=ClosedTipRotation(r.env,goal_deg=5. if extended else 2.,budget_steps=max(80,budget),max_attempts=2 if extended else 1)
    if mode=='rotation_contact':
        budget=min(135,remaining-70)
        obj.goal_deg=20. if abs(pose(r.env)[3][2])>np.deg2rad(100) else 5.
        obj.preferred_contact=contact
    if mode=='rotation_waypoint':
        name,residual=contact.rsplit(':',1)
        budget=min(105,remaining-95)
        obj.goal_deg=float(residual); obj.preferred_contact=name
    obj.budget_steps=budget; obj.closed=True; obj._phase('plan')
    if mode=='rotation_alt':
        from rotation_controller import pick_contact
        xy,yaw,goal,q=pose(r.env)
        first=pick_contact(xy,yaw,-q[2],r.obs['robot0_eef_pos'],goal,set(),{})
        if first is not None: obj.blocked.add(first[1].name)
    while not obj.done and len(r.actions)<LIMIT and valid(r.m):
        if ok(r.m,r.env): return r.release()
        if not r.step(obj.action(r.obs),('search','rotate_'+obj.phase)): return False
    if ok(r.m,r.env): return r.release()
    return r.prepare(True)

def configure_cache(r,args):
    r.cache=Checkpoints(r,getattr(args,'cache_nodes',0),getattr(args,'cache_audit_every',12))


def prefix_key(node):
    # Exact executed action prefix, independent of object identity or queue IDs.
    key=getattr(node,'_prefix_digest',None)
    if key is None:
        key=hashlib.sha256(np.asarray(node.actions,dtype=np.float64).tobytes()).digest()
        node._prefix_digest=key
    return key


def rescue_roots(roots,archive):
    """Cover early/intermediate states rather than repeatedly trimming late ones."""
    candidates={prefix_key(n):n for n in roots+archive if valid(n.metrics) and n.steps<=LIMIT-45}
    pool=list(candidates.values()); chosen={prefix_key(n):n for n in roots if valid(n.metrics) and n.steps<=LIMIT-45}
    for lo,hi in ((0,80),(80,150),(150,220),(220,280),(280,355)):
        subset=[n for n in pool if lo<=n.steps<hi]
        if subset:
            for n in sorted(subset,key=lambda n:float(loss(n.q))+.00025*n.steps)[:2]:chosen[prefix_key(n)]=n
    # Keep near-goal states when at least one contact can plausibly fit.
    for n in sorted(pool,key=lambda n:-n.metrics['iou'])[:3]:chosen[prefix_key(n)]=n
    return list(chosen.values())


def baseline(r,args):
    from dataset import stage_phase
    initial=r.start(); r.expected_initial=initial.sig.copy(); roots=[initial]
    configure_cache(r,args); r.cache.put(initial)
    expert=make_expert(r.env,args); best=initial; success=None
    recorder=DemoRecorder(); recorder.observe(r.obs,r.env,r.m,ok(r.m,r.env))
    while len(r.actions)<LIMIT:
        d=expert.delegate; eef=np.asarray(r.obs['robot0_eef_pos']); m=r.m
        # Capture only physically lifted, closed-gripper boundaries (no stale FSM state is copied).
        safe=bool(d is not None and d.grip_mode=='closed' and d.mode=='translate' and d.phase in ('plan','four_lift','release') and d._four_active is None and d._four_planned is None and eef[2]>=.896)
        handoff=bool(d is None and expert.rotator is not None and (expert.rotator.done or expert.rotator.phase=='plan') and expert.rotator.closed and eef[2]>=.896)
        if (safe or handoff) and valid(m) and LIMIT-len(r.actions)>=35:
            node=r.node(True,('baseline_prefix',))
            if not roots[-1].closed or np.linalg.norm(roots[-1].q-node.q)>.002 or abs(wrapped(roots[-1].q[2]-node.q[2]))>.035:
                roots.append(node); r.cache.put(node)
        a=expert.action(r.obs); tag=stage_phase(expert); recorder.action(a,*tag); r.step(a,tag); recorder.observe(r.obs,r.env,r.m,ok(r.m,r.env))
        if r.reached() and success is None:
            # Store exact first eligible prefix. Continue baseline to measure original 400-step outcome.
            success=r.node(True,('baseline',)); success_rec=recorder
            # Freeze recorder lists: subsequent baseline samples are not part of the demo.
            frozen=DemoRecorder()
            for k in ('obs','actions','phases','success','ious','target'): setattr(frozen,k,getattr(success_rec,k).copy())
            if getattr(args,'early_stop_baseline',False):break
        if r.m['iou']>best.metrics['iou']: best=r.node(True)
    final=r.node(False)
    return initial,roots,final,best,success,(frozen if success is not None else None)

def search(r,roots,args,priors,faces,log):
    """Best-first physical search across *all* depths, not a fixed beam layer.
    The queue contains remaining alternatives and newly reached states. A good
    late state can be expanded immediately instead of waiting for a whole layer.
    """
    best=max(roots,key=lambda n:n.metrics['iou']); trials=0; serial=itertools.count(); heap=[]; early_heap=[]; terminal_heap=[]; waypoint_heap=[]; seen=[]; terminal_pending=set(); waypoint_pending=set()
    rescue=r.rescue
    stage='targeted_rescue' if rescue else 'hybrid_first'
    def attempt_key(node,p): return (prefix_key(node),p.face,round(p.distance,4),p.mode)
    def enqueue_targeted(node,uid,include_waypoint=False):
        if not rescue:return
        key=prefix_key(node)
        for p in terminal_candidates(r.feedback,key,node,faces,r.env.target_xy,r.env.target_yaw,r.env.overlap_threshold):
            k=attempt_key(node,p)
            if k in r.search_tried or k in terminal_pending:continue
            terminal_pending.add(k)
            priority=p.estimate-.025*p.predicted_iou
            heapq.heappush(terminal_heap,(priority,next(serial),uid,node,p,0))
        if include_waypoint:
            for p in waypoint_proposals(node,r.env.target_xy,r.env.target_yaw,r.feedback,key):
                k=attempt_key(node,p)
                if k in r.search_tried or k in waypoint_pending:continue
                waypoint_pending.add(k)
                priority=p.estimate+.00020*node.steps
                heapq.heappush(waypoint_heap,(priority,next(serial),uid,node,p,0))
    def enqueue(node,root=False):
        if node.steps>=LIMIT-18 or not valid(node.metrics):return
        if any(old.closed==node.closed and old.steps<=node.steps+2 and np.linalg.norm(old.q[:2]-node.q[:2])<.003 and abs(wrapped(old.q[2]-node.q[2]))<.035 and np.linalg.norm(old.eef[:2]-node.eef[:2])<.018 for old in seen):return
        planning_node=node
        if rescue:
            key=prefix_key(node)
            local={k:r.feedback.rate(key,k,node.rates.get(k,priors[k])) for k in faces if r.feedback.for_face(key,k)}
            if local:planning_node=replace(node,rates={**node.rates,**local})
        opts=proposals(planning_node,r.env,priors,faces,max(14,args.branch) if rescue else args.branch)
        if rescue:
            if abs(node.q[2])>np.deg2rad(35) and LIMIT-node.steps>=155:
                for mode in ('rotation_multi','rotation_alt'):
                    opts.insert(0,Proposal(mode,0.,float(loss(node.q)*.55+.010),mode,predicted_steps=min(175,max(85,LIMIT-node.steps-85))))
            opts=explicit_rotations(node,r.env.target_xy,r.env.target_yaw)+opts
            opts=[p for p in opts if attempt_key(node,p) not in r.search_tried and (prefix_key(node),p.face) not in r.failed_approaches]
        if not opts and not rescue:return
        seen.append(node); uid=next(serial); r.search_archive.append(node)
        enqueue_targeted(node,uid,True)
        if r.cache is not None and not root: r.cache.put(node)
        for j,p in enumerate(opts):
            # Terminal estimate includes the predicted remaining trajectory.
            # Small penalties prefer states that actually improved and less
            # repetition. Give every baseline root an initial opportunity.
            priority=p.estimate+.10*float(loss(node.q))+.0018*j+.000035*node.steps
            if root and j==0:priority-=.016
            if rescue:
                # Reserve room for later corrections. Penalize consuming the
                # episode before alignment; keep near-goal short trims eligible.
                priority+=.00022*node.steps
                if root and j==0: priority-=.020
                if node.metrics['iou']>.65 and p.mode=='trim' and p.predicted_steps<=LIMIT-node.steps: priority-=.035
                if p.mode in ('rotation_multi','rotation_alt'): priority-=.025
                if p.mode=='rotation_contact': priority-=.055
                if node.steps>300 and node.metrics['iou']<.65: priority+=.045
            heapq.heappush(early_heap if rescue and node.steps<=240 else heap,(priority,next(serial),uid,node,p,j))
    # Closed pre-rotation root replaces the equivalent open initial state.
    closed=[n for n in roots if n.closed]
    start=roots if rescue else (closed if closed and closed[0].steps<=40 else roots)
    if rescue and closed and closed[0].steps<=40:
        start=closed  # the open root would only repeat the same close prefix
    for n in start:enqueue(n,True)
    while (heap or early_heap or terminal_heap or waypoint_heap) and trials<args.trials:
        # In rescue, deliberately allocate ~2/3 of available trials to early
        # roots; a single global queue otherwise over-selects near-deadline trim.
        queue=early_heap if early_heap and (not heap or trials%3!=2) else heap
        # Recovery alternates endpoint correction, waypoint rotation and normal planning.
        source="planner"
        if rescue:
            if terminal_heap and (trials%3==0 or not (queue or waypoint_heap)):
                queue=terminal_heap;source="endpoint_fit"
            elif waypoint_heap and (trials%3==1 or not queue):
                queue=waypoint_heap;source="rotation_waypoint"
            elif not queue and terminal_heap:
                queue=terminal_heap;source="endpoint_fit"
        _,_,uid,node,p,j=heapq.heappop(queue)
        if source=='endpoint_fit':terminal_pending.discard(attempt_key(node,p))
        if source=='rotation_waypoint':waypoint_pending.discard(attempt_key(node,p))
        if rescue and (prefix_key(node),p.face) in r.failed_approaches: continue
        if rescue:
            if attempt_key(node,p) in r.search_tried:continue
            if r.feedback.skip(prefix_key(node),p):r.feedback.pruned+=1;continue
        if rescue:
            key=prefix_key(node)
            stalled=p.mode in ('rotation_contact','rotation_waypoint') and rotation_stalled(r.feedback,key,p.face.split(':')[1])
            if stalled or saturated_trim(r.feedback,key,p):r.feedback.pruned+=1;continue
        restore_mode=r.cache.restore(node) if r.cache is not None else (r.replay(node) or 'replay')
        trials+=1; r.search_tried.add(attempt_key(node,p))
        common=dict(trial=len(log)+1,search_stage=stage,proposal_source=source,restore_mode=restore_mode,parent_id=uid,parent_steps=node.steps,face=p.face,distance_m=p.distance,mode=p.mode,tail='>'.join(f'{k}:{d:.3f}' for k,d in p.tail),estimated_total_steps=p.predicted_steps+node.steps,predicted_terminal_iou=p.predicted_iou,parent_iou=node.metrics['iou'])
        if not r.prepare(node.closed):
            log.append(dict(**common,steps=len(r.actions),iou=r.m['iou'],clear=False,success=False,reason='prepare_failed'));continue
        if ok(r.m,r.env) and not r.reached(): finish_contact(r)
        if r.reached():
            win=r.node(True,node.history+('settle',),node.rates);log.append(dict(**common,steps=win.steps,iou=win.metrics['iou'],clear=True,success=True,reason='settled'));return win,win,trials
        if not valid(r.m) or r.obs['robot0_eef_pos'][2]<.885:
            log.append(dict(**common,steps=len(r.actions),iou=r.m['iou'],clear=False,success=False,reason='not_clear'));continue
        before,heading,_,start_q=pose(r.env)
        good=(rotation_trial(r) if p.mode=='rotation' else rotation_trial(r,p.mode,p.face.split(':',1)[1]) if p.mode in ('rotation_contact','rotation_waypoint') else rotation_trial(r,p.mode)) if p.mode.startswith('rotation') else push(r,p,faces)
        after,ang,_,_=pose(r.env);rates=dict(node.rates);travel=0.
        if not p.mode.startswith('rotation'):
            delta=rot(-heading)@(after-before);normal=np.asarray(faces[p.face].normal);travel=float(delta@normal)
            if travel>.003:
                measured=np.r_[np.clip(delta/travel,-2,2),np.clip(wrapped(ang-heading)/travel,-18,18)]
                prior=np.asarray(rates.get(p.face,priors[p.face]));w=travel/(travel+.012);rates[p.face]=(1-w)*prior+w*measured
        child=r.node(True,node.history+(f'{p.face}:{p.distance:.4f}',),rates)
        if not good and not p.mode.startswith('rotation') and child.tags and child.tags[-1][1] in ('approach','descend') and np.linalg.norm(after-before)<.0005 and abs(wrapped(ang-heading))<.005:
            # Changing stroke length cannot fix the same failed approach.
            r.failed_approaches.add((prefix_key(node),p.face))
        r.feedback.add(prefix_key(node),Outcome(p.face,p.mode,p.distance,start_q.copy(),child.q.copy(),travel,child.steps-node.steps,bool(good),child.tags[-1][1] if child.tags else '',child.eef.copy()))
        log.append(dict(**common,steps=child.steps,iou=child.metrics['iou'],yaw_deg=child.metrics['yaw_error_degrees'],position_m=child.metrics['position_error'],clear=good,success=r.reached(),actual_travel_m=travel,actual_yaw_change_deg=np.rad2deg(wrapped(ang-heading)),parent_prefix=prefix_key(node).hex()[:16],end_phase=child.tags[-1][1] if child.tags else '',parent_error_x_m=node.q[0],parent_error_y_m=node.q[1],parent_error_yaw_deg=np.rad2deg(node.q[2]),end_error_x_m=child.q[0],end_error_y_m=child.q[1],end_error_yaw_deg=np.rad2deg(child.q[2])))
        if valid(child.metrics) and child.metrics['iou']>best.metrics['iou']:best=child
        if r.reached():return child,best,trials
        if rescue:
            enqueue_targeted(node,uid,p.mode.startswith('rotation'))
        # A real state can be useful even if its current IoU decreased. Do not
        # discard opposite-torque or repositioning moves needed by the tail.
        if good and valid(child.metrics) and child.steps>node.steps and (np.linalg.norm(child.q[:2]-node.q[:2])>.0015 or abs(wrapped(child.q[2]-node.q[2]))>.025):enqueue(child)
    return None,best,trials

def write_csv(path,rows):
    if not rows: return
    path=Path(path); keys=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.tmp');
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def record_selected_video(r,node,path,args,label,video_cls=None):
    """Render only the selected action prefix; publish only after replay verification."""
    if video_cls is None:
        from dataset import Video as video_cls
    path=Path(path); temp=path.with_name(path.stem+'.partial.mp4')
    video=None
    try:
        video=video_cls(temp,r.env,args)
        r.replay(node,frame=lambda step,m,tag,success: video.append(label,step,m,*tag,success))
        frames=video.frames
        video.close(); video=None
        if frames!=node.steps+1: raise RuntimeError(f'Incomplete video: {frames} frames')
        if not temp.is_file() or temp.stat().st_size==0: raise RuntimeError('Empty video')
        temp.replace(path)
        return str(path),frames,''
    except Exception as exc:
        if video is not None:
            try: video.close()
            except Exception: pass
        temp.unlink(missing_ok=True)
        return '',0,str(exc)

def run_case(a,out,seed,angle,record_video):
    """Independent environment per seed; worker never writes shared CSVs."""
    from dataset import make_env, exact_yaw_reset
    from dataset import reseed
    from geometry import FACE_BY_NAME
    model=load_model(a.response_model)
    # New environment per case, matching original physics and observation schema.
    video_error=''; video_path=''; video_frames=0; video_replay_steps=0
    try: env=make_env(record_video)
    except Exception as exc:
        if not record_video: raise
        video_error=str(exc); print(f'Offscreen unavailable: {exc}; continuing without video',flush=True); env=make_env(False)
    start=time.monotonic(); trial_log=[]; error=''; result=None; rec=None; count=0
    try:
        r=Runner(env,lambda:exact_yaw_reset(env,seed,angle,reseed),a.stable)
        init,roots,final,best,win,rec=baseline(r,a)
        if abs(np.rad2deg(init.q[2])+angle)>.05 or not valid(init.metrics):
            raise RuntimeError(f'Initial reset mismatch for seed {seed}')
        if win is not None: result=win; method='baseline'
        else:
            method='joint_lookahead'
            if a.trials:
                priors=dict(CalibratedPlanner('wide' if abs(angle)>60.05 else 'standard',model).PRIORS)
                for k,(base,turn) in EXTRA.items(): priors[k]=np.r_[priors[base][:2],turn]
                priors,faces=augmented(priors,FACE_BY_NAME)
                def attempt():
                    nonlocal best, result, rec, count, method
                    r.rescue=False; method='joint_lookahead'
                    result,search_best,count=search(r,roots,a,priors,faces,trial_log)
                    if search_best.metrics['iou']>best.metrics['iou']: best=search_best
                    if result is None and getattr(a,'rescue_trials',0)>0:
                        second=rescue_roots(roots,r.search_archive)
                        second=evidence_roots(second,r.search_archive,r.feedback,prefix_key)
                        if second:
                            r.rescue=True
                            rescue_args=SimpleNamespace(**vars(a)); rescue_args.trials=a.rescue_trials
                            result,search_best,extra=search(r,second,rescue_args,priors,faces,trial_log)
                            count+=extra
                            if result is not None: method='joint_lookahead_targeted'
                            if search_best.metrics['iou']>best.metrics['iou']: best=search_best
                    if result is not None:
                        rec=r.replay(result,record=True)
                        if not r.reached(): raise ReplayMismatch('Selected replay did not meet stable success and clearance')
                original_best=best
                try:
                    attempt()
                except (CacheMismatch,ReplayMismatch) as exc:
                    if r.cache is not None and (r.cache.backend is not None or r.cache.hits):
                        # Discard every cache-derived branch and repeat exactly
                        # from the original real baseline roots, not its children.
                        r.cache.disable(f'fallback to replay: {exc}')
                        for row in trial_log: row['discarded_cache_pass']=True
                        best=original_best; result=None; rec=None; r.search_archive=[];r.search_tried=set();r.failed_approaches=set();r.feedback=Feedback()
                        try: attempt()
                        except ReplayMismatch as replay_error: error=str(replay_error);result=None;rec=None
                    else: error=str(exc); result=None;rec=None
        saved=''; stats={}; demo_ok=False
        if result is not None and rec is not None:
            # Eligibility derives ONLY from recorded real environment states.
            if len(rec.actions)!=result.steps or not 1<=result.steps<=LIMIT or len(rec.obs)!=result.steps+1 or not all(rec.success[-a.stable:]): raise AssertionError('Invalid expert trajectory')
            stats=rec.stats(); stats['demo_prefix_steps']=result.steps
            name=f'seed_{seed}_{angle:+.4f}.npz'
            # Never overwrite previous successful data silently; unique output per run recommended.
            target=out/'demos'/name
            saved=rec.save(target,dict(seed=seed,initial_yaw_deg=angle,controller=method,strategy="hybrid",control_hz=20,action_dim=7,steps=result.steps,success_threshold=float(env.overlap_threshold),stable_steps=result.streak,final_iou=float(result.metrics['iou']),history=list(result.history),split='train_candidate' if a.collect else 'evaluation',calibration_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (a.rotation_model,a.response_model)})); demo_ok=True
        # Keep real baseline and selected/best failure traces for diagnosis.
        trace_dir=out/'traces'; trace_dir.mkdir(exist_ok=True)
        if final is not None: write_csv(trace_dir/f'{seed}_baseline.csv',final.trace)
        trace_node=result if demo_ok else best
        if trace_node.trace: write_csv(trace_dir/f'{seed}_selected_or_best.csv',trace_node.trace)
        row=dict(seed=seed,angle_deg=angle,angle_bin=band(angle),strategy="hybrid",baseline_final_iou=final.metrics['iou'] if final is not None else '',baseline_final_success=ok(final.metrics,env) if final is not None else '',baseline_stable=(win is not None) if final is not None else '',success=demo_ok,method=method if demo_ok else '',steps=result.steps if demo_ok else '',final_iou=result.metrics['iou'] if demo_ok else '',best_observed_iou=max(best.metrics['iou'],result.metrics['iou'] if demo_ok else 0),best_task_success=ok(trace_node.metrics,env),best_yaw_deg=trace_node.metrics['yaw_error_degrees'],best_position_m=trace_node.metrics['position_error'],search_trials=len(trial_log),physical_steps_including_replays=r.executed,demo_path=saved,error=error,wall_seconds=round(time.monotonic()-start,2),**stats)
        if record_video and not video_error:
            before=r.executed
            label=f'Seed {seed} | '+('selected' if demo_ok else 'best observed')
            video_path,video_frames,video_error=record_selected_video(r,trace_node,out/'videos'/f'seed_{seed}_{"success" if demo_ok else "best_failure"}.mp4',a,label)
            video_replay_steps=r.executed-before
            if video_error: print(f'Video error for {seed}: {video_error}',flush=True)
        row.update(**(r.cache.stats() if r.cache is not None else {}))
        row.update(adaptive_pruned=r.feedback.pruned, rescue_mode="targeted",baseline_evaluated_steps=final.steps if final is not None else '')
        row.update(total_wall_seconds=round(time.monotonic()-start,2),video_selected=record_video,video_path=video_path,video_frames=video_frames,video_error=video_error,video_replay_steps=video_replay_steps)
        return row, trial_log
    finally:
        if 'r' in locals() and r.cache is not None: r.cache.entries.clear()
        env.close()


def read_cases(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    result=[(int(r['seed']),float(r['angle_deg'])) for r in rows]
    if not result or len(set(s for s,_ in result))!=len(result) or any(not math.isfinite(x) or abs(x)>180 or s<0 for s,x in result):
        raise ValueError('Case manifest must contain unique nonnegative seeds and finite angles in [-180,180]')
    return result


def iter_case_results(a,out,selected,video_seeds,worker=run_case):
    """Yield completed seeds. Only the parent process aggregates shared results."""
    if a.workers==1:
        for seed,angle in selected:
            yield worker(a,out,seed,angle,seed in video_seeds)
    else:
        import os, multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
        original={key:os.environ.get(key) for key in keys}
        try:
            for key in keys: os.environ[key]='1'
            with ProcessPoolExecutor(max_workers=min(a.workers,len(selected)),mp_context=mp.get_context('spawn')) as pool:
                futures=[pool.submit(worker,a,out,seed,angle,seed in video_seeds) for seed,angle in selected]
                for future in as_completed(futures): yield future.result()
        finally:
            for key,value in original.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value


def main(root):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(); group.add_argument('--fresh',action='store_true'); group.add_argument('--collect',action='store_true')
    group.add_argument('--cases-file',type=Path,help='Run exact seed/angle pairs from an existing cases.csv')
    parser.add_argument('--workers',type=int,default=1,help='Independent spawned simulator processes; start with 2–4')
    parser.add_argument('--early-stop-baseline',action='store_true',help='Stop baseline at first stable successful prefix instead of continuing to step 400')
    parser.add_argument('--wide-angles',action='store_true',help='Sample only absolute initial yaw errors from 90 to 180 degrees; requires --fresh or --collect')
    parser.add_argument('--per-bin',type=int,default=None); parser.add_argument('--seed',type=int,default=None)
    parser.add_argument('--trials',type=int,default=48,help='Physical branch rollouts per failed case, not episode steps')
    parser.add_argument('--rescue-trials',type=int,default=24,help='Extra diverse trials only after the first search fails; 0 disables rescue')
    parser.add_argument('--cache-nodes',type=int,default=8,help='Bounded simulation checkpoint count per worker; 0 disables caching')
    parser.add_argument('--cache-audit-every',type=int,default=12,help='Compare cached continuation with full replay every N hits')
    parser.add_argument('--flush-every',type=int,default=8,help='Write aggregated CSVs every N completed cases, and at the end')
    parser.add_argument('--branch',type=int,default=8); parser.add_argument('--case-seeds',type=int,nargs='+',help='Restrict to seed IDs in the selected case set')
    parser.add_argument('--stable',type=int,default=10); parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--plan-only',action='store_true')
    vg=parser.add_mutually_exclusive_group(); vg.add_argument('--video',action='store_true',help='Record selected or best failure trajectory'); vg.add_argument('--no-video',action='store_true')
    parser.add_argument('--video-count',type=int,default=None,help='Record only N representative cases, chosen before testing; requires --video')
    parser.add_argument('--camera',default='agentview'); parser.add_argument('--width',type=int,default=640); parser.add_argument('--height',type=int,default=480)
    a=parser.parse_args(); a.fresh = a.fresh or not (a.collect or a.cases_file); a.rotation_model=str(root/'config/contact_model.json'); a.response_model=str(root/'config/contact_response_model.json')
    if min(a.width,a.height)<240 or a.width%16 or a.height%16: parser.error('Video dimensions must be multiples of 16, at least 240')
    if not(1<=a.stable<=40 and a.trials>=0 and a.branch>=3): parser.error('Invalid search budget')
    if a.video_count is not None and (not a.video or a.video_count<1): parser.error('--video-count requires --video and a positive count')
    if a.workers<1: parser.error('--workers must be positive')
    if not(0<=a.cache_nodes<=32 and a.cache_audit_every>=1 and a.rescue_trials>=0 and a.flush_every>=1): parser.error('Invalid cache/rescue/flush budget')
    if a.workers>1 and a.video: parser.error('Use --no-video for parallel collection; record representative cases separately with --workers 1')
    if a.wide_angles and not (a.fresh or a.collect): parser.error('--wide-angles requires --fresh or --collect')
    if a.per_bin is not None and (a.per_bin<2 or a.per_bin%2): parser.error('--per-bin must be positive and even')
    if a.cases_file:
        try: selected=read_cases(a.cases_file)
        except (OSError,ValueError,KeyError) as exc: parser.error(str(exc))
        if a.case_seeds:
            selected=[(s,x) for s,x in selected if s in a.case_seeds]
            if len(selected)!=len(set(a.case_seeds)): parser.error('Unknown seed in case manifest')
    elif a.fresh or a.collect:
        sampler=wide_cases if a.wide_angles else cases
        default_seed=(22000 if a.collect else 9200) if a.wide_angles else (12000 if a.collect else 7200)
        selected=sampler(a.seed if a.seed is not None else default_seed,a.per_bin or (40 if a.collect else 4),20261141 if a.collect else 20261132)
    if (a.fresh or a.collect) and a.case_seeds:
        selected=[(s,x) for s,x in selected if s in a.case_seeds]
        if len(selected)!=len(set(a.case_seeds)): parser.error('Unknown seed in sampled cases')
    video_seeds=typical_video_seeds(selected,a.video_count) if a.video else set()
    out=a.output or root/'results'/('collect' if a.collect else 'fresh' if a.fresh else 'cases')
    if a.wide_angles and a.output is None: out=out.with_name(out.name+'_90_180')
    out=Path(out)
    if (out/'summary.csv').exists() or (out/'demos').exists():
        if a.output is not None: parser.error('Output directory already contains results; choose a new --output')
        out=out.with_name(out.name+'_'+time.strftime('%Y%m%d_%H%M%S'))
        i=1
        while out.exists():
            out=out.with_name(out.name+f'_{i}'); i+=1
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'cases.csv',[dict(seed=s,angle_deg=x,angle_bin=band(x),video_selected=s in video_seeds) for s,x in selected])
    if a.video: print('Video selection: '+', '.join(f'{s} ({x:+.2f}°)' for s,x in selected if s in video_seeds),flush=True)
    (out/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2),encoding='utf-8')
    if a.plan_only: print(f'{len(selected)} cases: {out}/cases.csv'); return
    rows=[]; logs=[]; order={s:i for i,(s,_) in enumerate(selected)}
    def flush():
        write_csv(out/'summary.csv',sorted(rows,key=lambda x:order[x['seed']]))
        write_csv(out/'search_trials.csv',sorted(logs,key=lambda x:(order[x['seed']],x['trial'])))
    def accept(row,trial_log):
        seed=row['seed']; angle=row['angle_deg']
        rows.append(row); logs.extend(dict(seed=seed,**x) for x in trial_log)
        # One atomic per-case report protects completed work even before a batch flush.
        reports=out/'case_reports'; reports.mkdir(exist_ok=True)
        write_csv(reports/f'{seed}_summary.csv',[row]); write_csv(reports/f'{seed}_trials.csv',trial_log)
        if len(rows)%a.flush_every==0: flush()
        base=row['baseline_final_iou']; label=f'{base:.3f}' if base!='' else '-'
        print(f'{seed} {angle:+.2f}° | staged baseline={label} | best={row["best_observed_iou"]:.3f} | success={row["success"]} | steps={row["steps"] or "-"} | {row["method"] or row["error"] or "no successful rollout"} | {row["total_wall_seconds"]:.1f}s | video={row["video_path"] or "-"}',flush=True)
    batch_start=time.monotonic()
    try:
        for row,trial_log in iter_case_results(a,out,selected,video_seeds): accept(row,trial_log)
    finally: flush()
    print(f'Batch wall time: {time.monotonic()-batch_start:.1f}s',flush=True)
    n=sum(x['success'] for x in rows); print(f'Expert demos: {n}/{len(rows)}. Results: {out}',flush=True)
