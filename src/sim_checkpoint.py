"""Audited simulator checkpoints."""
from __future__ import annotations

import copy
import random
import numpy as np
from collections import OrderedDict


class CacheMismatch(RuntimeError):
    pass


# These Python objects have mutable control/sensor state. Native simulation,
# models, renderers, and XML objects are never deep-copied.
PREFIXES = ('robosuite.robots', 'robosuite.controllers', 'robosuite.utils.observables',
            'robosuite.utils.buffers', 'robosuite.models.grippers')


class PythonState:
    def __init__(self, env):
        self.objects = []
        self.memo = {}
        visited = set()
        def walk(x):
            if id(x) in visited: return
            visited.add(id(x))
            if isinstance(x, (str, bytes, int, float, bool, complex, type(None), np.generic, np.ndarray)):
                return
            if isinstance(x, (dict, list, tuple, set, frozenset)):
                values = list(x.items()) if isinstance(x, dict) else x
                for v in values: walk(v)
                return
            if isinstance(x, (np.random.Generator, np.random.RandomState)):
                # Deepcopy RNGs, preserving state and shared references.
                return
            self.memo[id(x)] = x
            if x is env or (hasattr(x, '__dict__') and type(x).__module__.startswith(PREFIXES)):
                self.objects.append(x)
                for v in vars(x).values(): walk(v)
        walk(env)
        # Preserve the object graph, including sensor closures and controller
        # references; copy values with a shared memo to retain Python aliases.
        self.values = copy.deepcopy([vars(o) for o in self.objects], dict(self.memo))
        self.np_random = copy.deepcopy(np.random.get_state())
        self.py_random = random.getstate()

    def restore(self):
        values = copy.deepcopy(self.values, dict(self.memo))
        for o, value in zip(self.objects, values):
            vars(o).clear(); vars(o).update(value)
        np.random.set_state(self.np_random); random.setstate(self.py_random)


def numeric_obs(obs):
    return {k: np.asarray(v).copy() for k,v in obs.items() if np.asarray(v).dtype.kind in 'biufc'}


def same_obs(a, b):
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape and np.allclose(a[k], b[k], rtol=1e-8, atol=1e-9, equal_nan=False) for k in a)


class NativeBackend:
    def __init__(self, env):
        import mujoco as mj
        self.mj = mj; self.env = env
        self.model = env.sim.model._model; self.data = env.sim.data._data
        self.spec = mj.mjtState.mjSTATE_INTEGRATION
        self.size = mj.mj_stateSize(self.model, self.spec)
        self.mode = 'full_mjdata' if callable(getattr(mj, 'mj_copyData', None)) else 'integration'
        self.native_bytes=int(getattr(self.data,'narena',0))+int(getattr(self.data,'nbuffer',0))
        # In a large-memory model prefer compact integration vectors.
        if self.native_bytes>32*1024*1024: self.mode='integration'
        # No native state can be shared between models or simulator instances.
        self._check()

    def _check(self):
        if self.env.sim.model._model is not self.model or self.env.sim.data._data is not self.data:
            raise CacheMismatch('Native simulator instance changed')

    def vector(self):
        self._check(); a=np.empty(self.size, dtype=np.float64)
        self.mj.mj_getState(self.model,self.data,a,self.spec)
        return a

    def save(self):
        self._check(); vector=self.vector(); full=None
        if self.mode=='full_mjdata':
            full=self.mj.MjData(self.model)
            self.mj.mj_copyData(full,self.model,self.data)
        # The target is stored in mjModel, not mjData.
        # MuJoCo may expose last-substep derived poses to the planner; forward
        # recomputes them at the new qpos. Preserve the exact boundary poses
        # for the compact backend. The next env.step forwards before control.
        derived={k:np.asarray(getattr(self.data,k)).copy() for k in ('xpos','xmat','xquat','site_xpos','site_xmat','geom_xpos','geom_xmat') if hasattr(self.data,k)} if full is None else {}
        return vector,full,self.model.body_pos.copy(),self.model.body_quat.copy(),derived,PythonState(self.env)

    def load(self, state):
        self._check(); vector,full,pos,quat,derived,python=state
        self.model.body_pos[:]=pos; self.model.body_quat[:]=quat
        if full is not None:
            self.mj.mj_copyData(self.data,self.model,full)
        else:
            self.mj.mj_setState(self.model,self.data,vector,self.spec)
            self.mj.mj_forward(self.model,self.data)
            # Forward can modify warm-start and other integration inputs.
            self.mj.mj_setState(self.model,self.data,vector,self.spec)
            for k,v in derived.items(): getattr(self.data,k)[:]=v
        python.restore(); self._check()
        if not np.array_equal(vector,self.vector()): raise CacheMismatch('Integration state differs after restore')


class Checkpoints:
    def __init__(self, runner, capacity=8, audit_every=12, backend=None):
        self.r=runner; self.capacity=capacity; self.audit_every=audit_every
        self.entries=OrderedDict(); self.hits=0; self.misses=0; self.audits=0; self.saved_steps=0; self.partial_replays=0; self.audit_steps=0; self.reason=''; self.backend=None
        if capacity:
            try: self.backend=backend or NativeBackend(runner.env)
            except Exception as exc: self.reason=f'cache unavailable: {type(exc).__name__}: {exc}'
        self.mode=getattr(self.backend,'mode','off')
        if self.mode=='full_mjdata' and getattr(self.backend,'native_bytes',0):
            self.capacity=min(self.capacity,max(1,128*1024*1024//self.backend.native_bytes))

    def put(self, node):
        if self.backend is None: return
        try:
            state=self.backend.save()
            self.entries[id(node)]=(node,state,numeric_obs(self.r.obs))
            self.entries.move_to_end(id(node))
            while len(self.entries)>self.capacity: self.entries.popitem(last=False)
        except Exception as exc:
            if self.hits: raise CacheMismatch(f'checkpoint capture failed: {exc}') from exc
            self.disable(f'checkpoint capture failed: {exc}')

    def disable(self, reason):
        self.reason=reason; self.entries.clear(); self.backend=None

    def _load(self, node, entry):
        from expert_search import signature
        r=self.r; _,state,obs=entry
        self.backend.load(state)
        r.obs={k:v.copy() for k,v in obs.items()};r.actions=node.actions.copy();r.tags=node.tags.copy();r.trace=node.trace.copy();r.streak=node.streak
        r.m=r.env.task_metrics()
        if hasattr(r.env,'timestep') and r.env.timestep!=node.steps: raise CacheMismatch('Environment timestep differs')
        if not np.allclose(signature(r.env,r.obs),node.sig,rtol=1e-8,atol=1e-9):
            raise CacheMismatch('Restored robot/object geometry differs')
        for k,v in node.metrics.items():
            if not np.isclose(r.m[k],v,rtol=1e-8,atol=1e-9): raise CacheMismatch(f'Restored task metric differs: {k}')

    def _probe(self,node):
        # Only transient diagnostic actions: never become part of a trajectory.
        r=self.r; commands=(np.r_[np.zeros(6),1. if node.closed else -1.],np.array([.04,-.03,.02,.01,0.,0.,1. if node.closed else -1.]))
        samples=[]
        for a in commands:
            if not r.step(a,('cache_audit','probe')): break
            samples.append((self.backend.vector(),numeric_obs(r.obs),dict(r.m),r.streak))
        return samples

    def restore(self,node):
        if self.backend is None: self.r.replay(node); return 'replay'
        entry=self.entries.get(id(node))
        if entry is None:
            self.misses+=1
            # Restore the longest resident exact ancestor, then replay only
            # the missing suffix. Keep the original logical step numbering.
            candidates=sorted((e[0] for e in self.entries.values() if 0<e[0].steps<node.steps),key=lambda n:-n.steps)
            ancestor=next((n for n in candidates if n.tags==node.tags[:n.steps] and np.array_equal(n.actions,node.actions[:n.steps])),None)
            if ancestor is None:
                self.r.replay(node);self.put(node);return 'replay'
            try:
                mode=self.restore(ancestor)
                for a,tag in zip(node.actions[ancestor.steps:],node.tags[ancestor.steps:]):
                    if not self.r.step(a,tag): raise CacheMismatch('Partial prefix exceeded episode budget')
                from expert_search import signature
                if not np.allclose(signature(self.r.env,self.r.obs),node.sig,rtol=1e-8,atol=1e-9) or self.r.streak!=node.streak:
                    raise CacheMismatch('Partial prefix did not reproduce target node')
                self.partial_replays+=1;self.put(node)
                return 'partial_'+mode
            except Exception as exc: raise CacheMismatch(str(exc)) from exc
        try:
            self._load(node,entry);self.entries.move_to_end(id(node));self.hits+=1
            # First restore and periodic audit compare actual continuation to
            # the original full-prefix replay, including controls and sensors.
            if self.hits==1 or self.hits%self.audit_every==0:
                audit_start=self.r.executed
                boundary=(self.backend.vector(),numeric_obs(self.r.obs)); before=self._probe(node);self.r.replay(node)
                if not np.allclose(boundary[0],self.backend.vector(),rtol=1e-8,atol=1e-9) or not same_obs(boundary[1],numeric_obs(self.r.obs)):
                    raise CacheMismatch('Restored boundary differs from full replay')
                # Capture replay's exact parent before doing the same probe.
                self.put(node);fresh=self.entries[id(node)];after=self._probe(node)
                if len(before)!=len(after): raise CacheMismatch('Probe lengths differ')
                for (x,o,m,s),(y,p,n,t) in zip(before,after):
                    if not np.allclose(x,y,rtol=1e-8,atol=1e-9) or not same_obs(o,p) or s!=t or any(not np.isclose(m[k],n[k],rtol=1e-8,atol=1e-9) for k in m):
                        raise CacheMismatch('Action-continuation audit disagrees with full replay')
                self._load(node,fresh);self.audits+=1;self.audit_steps+=self.r.executed-audit_start-node.steps
                return 'audited'
            self.saved_steps+=node.steps
            return 'cache'
        except Exception as exc:
            raise CacheMismatch(str(exc)) from exc

    def stats(self):
        return dict(cache_backend=self.mode,cache_active=self.backend is not None,cache_fallback=self.reason.startswith('fallback to replay:'),cache_hits=self.hits,cache_misses=self.misses,cache_audits=self.audits,cache_prefix_steps_saved=self.saved_steps,cache_note=self.reason,cache_capacity=self.capacity,cache_partial_replays=self.partial_replays,cache_extra_audit_steps=self.audit_steps)
