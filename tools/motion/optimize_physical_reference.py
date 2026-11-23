"""Contact-aware trajectory optimization, NOT PPO/AMP training.

The teacher optimizes 50 Hz position targets and integrates the unmodified robot
with native effort-limited PD, free root and ground contact at 1 kHz. Only actual
physical states can become a demonstration, and only after a separate audit.

A nominal torque seed is merely an optimizer initialization, never a certificate.
No shared environment/package/robot asset/training task is modified by this tool.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast
import numpy as np
import mujoco as _mj
from mujoco import rollout
from scipy.spatial.transform import Rotation
from play_retarget_motion import load_motion, joint_mapping, MODEL_PATH
from physical_reference_model import build_physical_model
mj = cast(Any, _mj)
ROOT = Path(__file__).resolve().parents[2]

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--seed', type=Path, required=True, help='Hashed nominal torque seed; not an accepted physical trajectory')
    p.add_argument('--out', type=Path, required=True, help='New output directory; refuses overwrite')
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--initial-controls',type=Path,default=None,help='Previous action sequence used only as a numerical warm start')
    p.add_argument('--substep-speed-weight',type=float,default=5000.)
    p.add_argument('--duration', type=float, default=12.0)
    p.add_argument('--horizon', type=int, default=60)
    p.add_argument('--iters', type=int, default=25)
    p.add_argument('--replan', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--quality',action='store_true',help='Include stance velocity, foot orientation and intermediate 1 kHz speed costs')
    p.add_argument('--contact-time-constant',type=float,default=None,help='Explicit in-memory contact calibration; never changes asset')
    args = p.parse_args()
    if not 5 <= args.horizon <= 100 or not 1 <= args.iters <= 100 or (not 1 <= args.replan <= args.horizon) or (not 1 <= args.threads <= 4) or (not np.isfinite(args.duration)) or (args.duration <= 0):
        p.error('Invalid CPU optimizer parameters')
    for path in [args.motion,args.seed,args.out,args.resume,args.initial_controls]:
        if path is not None and 'lilin' in str(path.resolve()).lower():
            p.error('Another user project is outside scope')
    if args.out.exists():
        p.error('Output exists: refusing overwrite')
    return args

def run(args: argparse.Namespace) -> None:
    (a, names, dt) = load_motion(args.motion)
    (m,ids,adr,dofs,bids,sole_vertices,anchors,kp,kd,effort)=build_physical_model(names,args.contact_time_constant)
    nv,nq,nu=m.nv,m.nq,13
    ns=2*nv
    stride=round(dt/m.opt.timestep)
    if abs(stride*m.opt.timestep-dt)>1e-10:
        raise ValueError('Reference dt must be a multiple of 1 ms')
    n = len(a['time_s'])
    Q = np.tile(m.qpos0, (n, 1))
    Q[:, :3] = a['root_positions_m']
    Q[:, 3:7] = a['root_quaternions_xyzw'][:, [3, 0, 1, 2]]
    Q[:, adr] = a['joint_positions_rad']
    V = np.zeros((n, nv))
    for i in range(n):
        (l, h) = (max(0, i - 1), min(n - 1, i + 1))
        mj.mj_differentiatePos(m, V[i], (h - l) * dt, Q[l], Q[h])
    seed = np.load(args.seed, allow_pickle=False)
    if str(seed['reference_sha256'].item()) != hashlib.sha256(args.motion.read_bytes()).hexdigest():
        raise ValueError('Optimizer seed belongs to a different motion')
    FF = seed['torque_seed_Nm'].copy()
    if FF.shape != (n, 13) or not np.isfinite(FF).all():
        raise ValueError('Invalid optimizer seed')
    Uref = Q[:, adr] + (FF + kd * V[:, dofs]) / kp
    Uinitial=Uref.copy()
    if args.initial_controls is not None:
        with np.load(args.initial_controls,allow_pickle=False) as previous:
            warm=previous['controls']
        if warm.ndim!=2 or warm.shape[1]!=13 or not np.isfinite(warm).all():
            raise ValueError('Invalid initial control sequence')
        count=min(len(Uinitial),len(warm));Uinitial[:count]=warm[:count]
    refd = mj.MjData(m)
    work = mj.MjData(m)
    jp = np.zeros((3, nv))
    jr = np.zeros_like(jp)
    refpoint = np.zeros((n, 2, 3))
    refrot = np.zeros((n, 2, 3, 3))
    for i in range(n):
        refd.qpos[:] = Q[i]
        refd.qvel[:] = V[i]
        mj.mj_forward(m, refd)
        for (j, b) in enumerate(bids):
            refrot[i, j] = refd.xmat[b].reshape(3, 3)
            refpoint[i, j] = refd.xpos[b] + refrot[i, j] @ anchors[j]
    workers = [mj.MjData(m) for _ in range(args.threads)]
    pool = cast(Any,rollout.Rollout(nthread=args.threads))

    def integrate_state(s, dx):
        out = s.copy()
        mj.mj_integratePos(m, out[1:1 + nq], dx[:nv], 1)
        out[1 + nq:] += dx[nv:]
        return out

    def diff(s1, s2):
        out = np.zeros(ns)
        mj.mj_differentiatePos(m, out[:nv], 1, s1[1:1 + nq], s2[1:1 + nq])
        out[nv:] = s2[1 + nq:] - s1[1 + nq:]
        return out

    def path_residual(path):
        if not args.quality:
            return np.zeros(path.shape[:-2]+(1,))
        # Penalize EVERY physics substep, not just the 50 Hz endpoints.
        velocities=path[...,1+nq+dofs]
        return (np.sqrt(args.substep_speed_weight/stride)*np.maximum(np.abs(velocities)-4.,0.)).reshape(path.shape[:-2]+(-1,))

    def simulate(s, u, with_path=False) -> Any:
        (states, _) = pool.rollout(m, workers, s[None], np.repeat(u[None, None, :], stride, axis=1))
        end = states[0, -1]
        if not np.isfinite(end).all() or abs(end[0] - s[0] - dt) > 1e-08 or np.max(np.abs(end[1 + nq:])) > 200:
            raise FloatingPointError('Invalid physics transition; numerical reset is not allowed')
        return (end,states[0]) if with_path else end

    def derivatives(states, U):
        H = len(U)
        eps = 2e-05
        batch = H * (ns + nu + 1)
        S = np.repeat(states[:-1], ns + nu + 1, axis=0)
        C = np.repeat(U, ns + nu + 1, axis=0)
        for k in range(H):
            off = k * (ns + nu + 1)
            for j in range(ns):
                dx = np.zeros(ns)
                dx[j] = eps
                S[off + 1 + j] = integrate_state(S[off], dx)
            C[off + 1 + ns:off + 1 + ns + nu] += eps * np.eye(nu)
        (result, _) = pool.rollout(m, workers, S, np.repeat(C[:, None, :], stride, axis=1))
        F = result[:, -1]
        As = np.empty((H, ns, ns))
        Bs = np.empty((H, ns, nu))
        residuals=path_residual(result)
        extra=[]
        for k in range(H):
            off = k * (ns + nu + 1)
            r0=residuals[off]
            Jres=(residuals[off+1:off+1+ns+nu]-r0).T/eps
            extra.append((Jres.T@r0,Jres.T@Jres))
            base = F[off]
            for j in range(ns):
                As[k, :, j] = diff(base, F[off + 1 + j]) / eps
            for j in range(nu):
                Bs[k, :, j] = diff(base, F[off + 1 + ns + j]) / eps
        return (As, Bs,extra)
    Wq = np.r_[[250, 400, 1500], [700, 700, 200], [4.0] * 13]
    Wv = np.r_[[15, 15, 30], [6, 6, 3], [0.1] * 13]
    Wr = np.r_[Wq, Wv]
    wu = 0.8
    lim = m.jnt_range[ids]
    mid = lim.mean(1)
    halfrange = (lim[:, 1] - lim[:, 0]) * 0.45
    lo = mid - halfrange
    hi = mid + halfrange

    def cost(s, u, k, terminal=False, deriv=False) -> Any:
        k = min(k, n - 1)
        q = s[1:1 + nq]
        v = s[1 + nq:]
        e = np.zeros(nv)
        mj.mj_differentiatePos(m, e, 1, Q[k], q)
        err = np.r_[e, v - V[k]]
        cost = 0.5 * np.dot(Wr, err ** 2)
        gx = Wr * err
        H = np.diag(Wr)
        gu = np.zeros(nu)
        Hu = np.eye(nu) * wu
        if u is not None:
            du = u - Uref[k]
            cost += 0.5 * wu * np.dot(du, du)
            gu = wu * du
        work.qpos[:] = q
        work.qvel[:] = v
        mj.mj_forward(m, work)
        for (j, b) in enumerate(bids):
            r = work.xmat[b].reshape(3, 3)
            pt = work.xpos[b] + r @ anchors[j]
            delta = pt - refpoint[k, j]
            mj.mj_jac(m, work, jp, jr, pt, b)
            weight = np.array([1600, 2000, 4000]) if a['contact_candidates'][k, j] else np.array([800, 1000, 3000])
            J = np.c_[jp, np.zeros_like(jp)]
            cost += 0.5 * np.dot(weight, delta ** 2)
            gx += J.T @ (weight * delta)
            H += J.T @ (weight[:, None] * J)
            if args.quality:
                # Feet should face the adapted reference; no fabricated arm/spine joints.
                orient=Rotation.from_matrix(r@refrot[k,j].T).as_rotvec()
                Jrot=np.c_[jr,np.zeros_like(jr)]
                wrot=50.
                cost+=.5*wrot*np.dot(orient,orient)
                gx+=wrot*Jrot.T@orient
                H+=wrot*Jrot.T@Jrot
                if a['contact_candidates'][k,j]:
                    velocity=jp@v
                    Jvel=np.c_[np.zeros_like(jp),jp]
                    cost+=.5*120.*np.dot(velocity,velocity)
                    gx+=120.*Jvel.T@velocity
                    H+=120.*Jvel.T@Jvel
        for (j, idx) in enumerate(dofs):
            ev = max(abs(v[idx]) - 4.5, 0)
            sg = np.sign(v[idx])
            cost += 100 * ev ** 2
            if ev > 0:
                gx[nv + idx] += 200 * ev * sg
                H[nv + idx, nv + idx] += 200
            eq = q[adr[j]] - np.clip(q[adr[j]], lo[j], hi[j])
            cost += 300 * eq ** 2
            if eq:
                gx[idx] += 600 * eq
                H[idx, idx] += 600
        return (cost, gx, H, gu, Hu) if deriv else cost

    def optimize(s0, U, k0, maxit):
        H = len(U)
        reg = 10.0
        X = [s0.copy()]
        pathcost=0.
        for u in U:
            endpoint,path=simulate(X[-1],u,with_path=True)
            pathcost+=.5*np.sum(path_residual(path)**2)
            X.append(endpoint)
        X = np.asarray(X)
        best = pathcost+sum((cost(X[k], U[k], k0 + k) for k in range(H))) + cost(X[-1], None, k0 + H)
        completed = 0
        for it in range(maxit):
            (As, Bs,extras) = derivatives(X, U)
            cs = [cost(X[k], U[k], k0 + k, deriv=True) for k in range(H)]
            (_, vx, vxx, _, _) = cost(X[-1], None, k0 + H, deriv=True)
            gains = np.zeros((H, nu, ns))
            ks = np.zeros((H, nu))
            try:
                for k in range(H - 1, -1, -1):
                    (_, lx, lxx, lu, luu) = cs[k]
                    exg,exH=extras[k]
                    lx=lx+exg[:ns];lu=lu+exg[ns:]
                    lxx=lxx+exH[:ns,:ns];luu=luu+exH[ns:,ns:]
                    A = As[k]
                    B = Bs[k]
                    qx = lx + A.T @ vx
                    qu = lu + B.T @ vx
                    qxx = lxx + A.T @ vxx @ A
                    quu = luu + B.T @ vxx @ B
                    qux = B.T @ vxx @ A+exH[ns:,:ns]
                    L = np.linalg.cholesky(quu + reg * np.eye(nu))
                    K = -np.linalg.solve(L.T, np.linalg.solve(L, qux))
                    kk = -np.linalg.solve(L.T, np.linalg.solve(L, qu))
                    gains[k] = K
                    ks[k] = kk
                    vx = qx + K.T @ quu @ kk + K.T @ qu + qux.T @ kk
                    vxx = qxx + K.T @ quu @ K + K.T @ qux + qux.T @ K
                    vxx = (vxx + vxx.T) / 2
            except np.linalg.LinAlgError:
                reg *= 10
                continue
            accept = False
            for alpha in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                newX = [s0.copy()]
                newU = []
                value = 0
                for k in range(H):
                    delta = diff(X[k], newX[-1])
                    u = U[k] + alpha * ks[k] + gains[k] @ delta
                    u = np.clip(u, lo - 1.5, hi + 1.5)
                    value += cost(newX[-1], u, k0 + k)
                    newU.append(u)
                    try:
                        endpoint,path=simulate(newX[-1],u,with_path=True)
                        value+=.5*np.sum(path_residual(path)**2)
                        newX.append(endpoint)
                    except FloatingPointError:
                        break
                    if not np.isfinite(newX[-1]).all() or value > best * 2:
                        break
                if len(newU) != H or len(newX) != H + 1:
                    continue
                value += cost(newX[-1], None, k0 + H)
                if value < best:
                    improvement = best - value
                    X = np.asarray(newX)
                    U = np.asarray(newU)
                    best = value
                    reg = max(reg / 3, 0.001)
                    accept = True
                    completed += 1
                    break
            if not accept:
                reg *= 10
                if reg > 10000000.0:
                    break
            elif improvement < 0.0001:
                break
        return (U, X, best, completed)
    s = np.r_[0, Q[0], V[0]]
    s[3] -= 0.00015
    N = min(round(args.duration / dt), n - 1)
    H = args.horizon
    U = Uinitial[:H].copy()
    states = [s.copy()]
    controls = []
    log = []
    wall = time.time()
    start = 0
    if args.resume is not None:
        prior = np.load(args.resume, allow_pickle=False)
        states = list(prior['states'])
        controls = list(prior['controls'])
        log = list(prior['log'])
        start = len(controls)
        s = states[-1].copy()
        if len(states) != start + 1 or not np.allclose(np.array(states)[:, 0], np.arange(start + 1) * dt, atol=1e-08):
            raise ValueError('Resume contains resets or inconsistent times')
        U = prior['future_controls'].copy() if 'future_controls' in prior else Uinitial[start:start + H].copy()
        if start % args.replan:
            raise ValueError('Resume must be at a replanning boundary')
    args.out.mkdir(parents=True, exist_ok=False)
    meta = {'status': 'EXPERIMENTAL_IN_PROGRESS_NOT_APPROVED', 'input_reference': str(args.motion.resolve()), 'input_reference_sha256': hashlib.sha256(args.motion.read_bytes()).hexdigest(), 'optimizer_seed_sha256': hashlib.sha256(args.seed.read_bytes()).hexdigest(), 'mjcf_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), 'physics_dt_s': 0.001, 'control_dt_s': dt, 'horizon_frames': H, 'iterations_per_replan': args.iters, 'replan_frames': args.replan, 'threads': args.threads, 'resume_from': str(args.resume) if args.resume else None, 'gravity': m.opt.gravity.tolist(), 'joint_names': names, 'kp': kp.tolist(), 'kd': kd.tolist(), 'effort_limits_Nm': effort.tolist(), 'no_root_forces': True, 'no_physical_state_resets': True, 'no_learning': True, 'amp_integrated': False,'initial_controls_sha256':sha256_file(args.initial_controls) if args.initial_controls else None,'substep_speed_weight':args.substep_speed_weight,'intermediate_speed_cost':args.quality,'contact_time_constant_s':args.contact_time_constant,'optimizer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out / 'run_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    for k in range(start, N):
        if k % args.replan == 0:
            h = min(H, n - 1 - k)
            if k > start:
                U = np.vstack([U[args.replan:], Uinitial[k + len(U) - args.replan:k + len(U)]])[:h]
            if len(U) < h:
                U = np.vstack([U, Uinitial[k + len(U):k + h]])
            (U, XX, value, completed) = optimize(s, U, k, args.iters if k > start else args.iters * 2)
            q = s[1:1 + nq]
            v = s[1 + nq:]
            tilt = np.rad2deg(np.arccos(np.clip(Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1)))
            print('MPC', k, round(k * dt, 3), 'cost', round(value, 2), 'ok', completed, 'height', round(q[2], 3), 'tilt', round(tilt, 2), 'vmax', round(max(abs(v[dofs])), 2), 'wall', round(time.time() - wall, 1), flush=True)
            log.append([k, k * dt, value, completed, q[2], tilt, max(abs(v[dofs]))])
        u = U[k % args.replan]
        s = simulate(s, u)
        states.append(s.copy())
        controls.append(u.copy())
        if s[3] < 0.35 or not np.isfinite(s).all():
            print('FALL', k, flush=True)
            break
        if (k + 1) % args.replan == 0:
            future = np.vstack([U[args.replan:], Uinitial[k + 1 + len(U) - args.replan:k + 1 + len(U)]])[:min(H, n - 1 - (k + 1))]
            np.savez_compressed(args.out / 'progress.npz', states=states, controls=controls, log=log, future_controls=future)
    np.savez_compressed(args.out / 'rollout.npz', states=states, controls=controls, log=log)
    meta.update(status='OPTIMIZATION_FINISHED_REQUIRES_INDEPENDENT_AUDIT', completed_frames=len(controls), wall_time_s=time.time() - wall)
    (args.out / 'run_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print('DONE', len(controls) * dt, 'wall', time.time() - wall, flush=True)
    pool.close()

def main():
    run(parse_args())
if __name__ == '__main__':
    main()
