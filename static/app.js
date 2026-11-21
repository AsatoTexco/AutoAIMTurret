/* ============================================================================
 * AutoAIM Turret — Unified Single-Flight Controller, Mobile Game HUD & Eco CV
 * ============================================================================ */

const TARGET_NAMES = { person: 'PESSOA', dog: 'CACHORRO', face: 'ROSTO' };
const RES_NAMES = {
  0: 'QVGA (320×240)',
  1: 'CIF (400×296)',
  2: 'VGA (640×480)',
  3: 'SVGA (800×600)',
};
const CAM_GAIN_NAMES = {
  0: '2x',
  1: '4x',
  2: '8x',
  3: '16x',
  4: '32x',
  5: '64x',
};

/**
 * Single-Flight HTTP dispatcher for real-time GET /servo (x, y, laser, led only).
 * Camera settings (res, bri, ae, gain, night) use separate GET /cam calls.
 *
 * - Sends only changed parameters (preserving ESP32 500 ms servo Auto-Detach when toggling Laser/Flash).
 * - Fires ONLY when at least one parameter actually changed.
 * - Priority:
 *     * 0 ms (immediate) when led or laser changes
 *     * 30 ms minimum interval when only x/y changes
 */
class SingleFlightController {
  constructor({ endpoint = '/servo', moveIntervalMs = 30, onDispatch = null } = {}) {
    this.endpoint = endpoint;
    this.moveIntervalMs = moveIntervalMs;
    this.onDispatch = onDispatch;

    this.desired = { x: 90, y: 90, led: 0, laser: 0 };
    this.sent = { x: 90, y: 90, led: 0, laser: 0 };
    this.sendAngles = false;

    this.inFlight = false;
    this.timerId = null;
    this.lastSendAt = 0;
    this.lastRttMs = null;
    this.lastUrl = '/servo?x=90&y=90&laser=0&led=0';
    this.lastPriority = false;
    this.errorBackoffMs = 250;
    this.maxErrorBackoffMs = 2000;
    this.consecutiveErrors = 0;
    this.retryAfterTs = 0;
  }

  /** Sync local baseline from server telemetry without triggering a network send. */
  syncFromServer({ x, y, led, laser }) {
    if (this.inFlight || this.hasPending()) return;
    if (typeof x === 'number') {
      this.desired.x = Math.round(x);
      this.sent.x = Math.round(x);
    }
    if (typeof y === 'number') {
      this.desired.y = Math.round(y);
      this.sent.y = Math.round(y);
    }
    if (typeof led === 'boolean' || typeof led === 'number') {
      const v = led ? 1 : 0;
      this.desired.led = v;
      this.sent.led = v;
    }
    if (typeof laser === 'boolean' || typeof laser === 'number') {
      const v = laser ? 1 : 0;
      this.desired.laser = v;
      this.sent.laser = v;
    }
  }

  update(patch = {}) {
    if (patch.x !== undefined) {
      this.desired.x = Math.max(0, Math.min(180, Math.round(patch.x)));
      this.sendAngles = true;
    }
    if (patch.y !== undefined) {
      this.desired.y = Math.max(0, Math.min(180, Math.round(patch.y)));
      this.sendAngles = true;
    }
    if (patch.includeAngles !== undefined) {
      this.sendAngles = Boolean(patch.includeAngles);
    }
    if (patch.led !== undefined) {
      this.desired.led = patch.led ? 1 : 0;
    }
    if (patch.laser !== undefined) {
      this.desired.laser = patch.laser ? 1 : 0;
    }
    this.schedule();
  }

  isPriorityDirty() {
    return (
      this.desired.led !== this.sent.led ||
      this.desired.laser !== this.sent.laser
    );
  }

  isMoveDirty() {
    return (
      this.sendAngles &&
      (this.desired.x !== this.sent.x || this.desired.y !== this.sent.y)
    );
  }

  hasPending() {
    return this.isPriorityDirty() || this.isMoveDirty();
  }

  schedule() {
    const priorityDirty = this.isPriorityDirty();
    const moveDirty = this.isMoveDirty();

    if (!priorityDirty && !moveDirty) {
      return;
    }

    if (this.inFlight) {
      return;
    }

    const now = performance.now();
    if (this.retryAfterTs > now) {
      if (this.timerId === null) {
        const waitMs = Math.max(1, Math.ceil(this.retryAfterTs - now));
        this.timerId = setTimeout(() => {
          this.timerId = null;
          this.schedule();
        }, waitMs);
      }
      return;
    }

    if (priorityDirty) {
      if (this.timerId !== null) {
        clearTimeout(this.timerId);
        this.timerId = null;
      }
      this.dispatch(true);
      return;
    }

    const elapsed = performance.now() - this.lastSendAt;
    if (elapsed >= this.moveIntervalMs) {
      if (this.timerId !== null) {
        clearTimeout(this.timerId);
        this.timerId = null;
      }
      this.dispatch(false);
    } else if (this.timerId === null) {
      const waitMs = Math.max(1, Math.ceil(this.moveIntervalMs - elapsed));
      this.timerId = setTimeout(() => {
        this.timerId = null;
        this.schedule();
      }, waitMs);
    }
  }

  async dispatch(isPriority) {
    if (this.inFlight) return;

    const snap = {
      x: this.desired.x,
      y: this.desired.y,
      led: this.desired.led,
      laser: this.desired.laser,
      includeAngles: this.sendAngles,
    };

    const anglesDirty =
      snap.includeAngles && (snap.x !== this.sent.x || snap.y !== this.sent.y);
    const ledDirty = snap.led !== this.sent.led;
    const laserDirty = snap.laser !== this.sent.laser;

    const params = new URLSearchParams();
    if (anglesDirty) {
      params.set('x', String(snap.x));
      params.set('y', String(snap.y));
    }
    if (anglesDirty || laserDirty) {
      params.set('laser', String(snap.laser));
    }
    if (anglesDirty || ledDirty) {
      params.set('led', String(snap.led));
    }
    const qs = params.toString();
    if (!qs) {
      return;
    }

    const url = `${this.endpoint}?${qs}`;
    this.inFlight = true;
    this.lastSendAt = performance.now();
    this.lastPriority = isPriority;
    this.lastUrl = url;

    const t0 = performance.now();
    let succeeded = false;
    try {
      const response = await fetch(url, { method: 'GET', cache: 'no-store' });
      const rtt = performance.now() - t0;
      if (response.ok) {
        succeeded = true;
        this.consecutiveErrors = 0;
        this.retryAfterTs = 0;
        if (anglesDirty) {
          this.sent.x = snap.x;
          this.sent.y = snap.y;
        }
        if (anglesDirty || ledDirty) {
          this.sent.led = snap.led;
        }
        if (anglesDirty || laserDirty) {
          this.sent.laser = snap.laser;
        }
        this.lastRttMs = rtt;
        if (this.onDispatch) {
          this.onDispatch({ url, rttMs: rtt, isPriority, snap });
        }
      }
    } catch (_err) {
      // Transient network error; next state update or retry will converge
    } finally {
      if (!succeeded) {
        this.consecutiveErrors = Math.min(this.consecutiveErrors + 1, 5);
        const backoff = Math.min(
          this.maxErrorBackoffMs,
          this.errorBackoffMs * Math.pow(2, this.consecutiveErrors - 1)
        );
        this.retryAfterTs = performance.now() + backoff;
      }
      this.inFlight = false;
      if (this.hasPending()) {
        this.schedule();
      }
    }
  }
}

/* ================= APPLICATION STATE ================= */
const state = {
  initialized: false,
  autoAim: true,
  aimMode: 'center',
  rotation: 90,
  resolution: 3,
  camBri: 1,
  camAe: 3,
  camGain: 5,
  camNight: true,
  lastCamSentAt: 0,
  autoFireLaser: true,
  autoFireLed: false,
  invertX: false,
  invertY: false,

  // Hardware control timing & last payload
  hwMoveIntervalMs: 30,
  hwLastPayload: '',

  // Eco mode
  ecoMode: true,
  ecoDelayS: 15,
  ecoFps: 1.0,
  ecoActive: false,
  ecoMotionWoken: false,
  ecoMotionWake: true,
  ecoMotionFlash: true,
  ecoFlashDarkOnly: false,
  ecoMotionSensitivity: 50,
  ecoFlashDurationS: 6,

  // Discord face log
  discordConfigured: false,
  discordEnabled: true,
  discordCooldownS: 30,
  discordAbsenceS: 15,
  discordIncludeScene: true,
  discordLogDogs: false,

  // Local smooth manual pose (floating-point for sub-degree FPS / touch accumulation)
  manualX: 90.0,
  manualY: 90.0,
  lastManualInputAt: 0,

  // Independent latch + multi-touch hold states
  latchLaser: false,
  holdLaserBtn: false,
  mouseLaserHeld: false,

  latchLed: false,
  holdLedBtn: false,
  mouseLedHeld: false,

  // Desktop Pointer Lock FPS state
  pointerLocked: false,
  mouseSensitivity: parseFloat(localStorage.getItem('autoaim_mouse_sens') || '0.18') || 0.18,

  // Mobile / Fullscreen Game Cockpit state
  gameMode: false,
  gameDrawerOpen: false,
  enteredNativeFs: false,
  savedScrollY: 0,
  lastConfigSentAt: 0,
  joystickSpeedDegPerSec: parseFloat(localStorage.getItem('autoaim_joy_sens') || '115') || 115,
  touchAimSensitivity: parseFloat(localStorage.getItem('autoaim_touch_sens') || '0.24') || 0.24,
  targetLocked: false,
};

const activeLaserPointers = new Set();
const activeLedPointers = new Set();

const joystickState = {
  active: false,
  pointerId: null,
  padEl: null,
  knobEl: null,
  normX: 0,
  normY: 0,
  rafId: null,
  lastFrameTs: 0,
};

const swipeAimState = {
  active: false,
  pointerId: null,
  lastX: 0,
  lastY: 0,
};

const sf = new SingleFlightController({
  endpoint: '/servo',
  moveIntervalMs: 30,
  onDispatch: ({ url, rttMs, isPriority }) => {
    const roundedRtt = `${Math.round(rttMs)} ms`;
    const elProto = document.getElementById('protoPreview');
    if (elProto) elProto.textContent = `GET ${url}`;
    const elRtt = document.getElementById('mRtt');
    if (elRtt) elRtt.textContent = roundedRtt;
    const elGameRtt = document.getElementById('gameHudRtt');
    if (elGameRtt) elGameRtt.textContent = `${Math.round(rttMs)}ms`;
    const elTag = document.getElementById('mPriorityTag');
    if (elTag) {
      elTag.textContent = isPriority
        ? '0ms IMEDIATO (Laser/Flash)'
        : `${state.hwMoveIntervalMs}ms Movimento`;
    }
  },
});

/* ================= HAPTIC & TOAST HELPERS ================= */
function triggerHaptic(pattern = 10) {
  if (typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function') {
    try {
      navigator.vibrate(pattern);
    } catch (_err) {
      // ignore unsupported platforms
    }
  }
}

let toastTimer = null;
function showToast(msg, durationMs = 2200) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.classList.remove('show');
  }, durationMs);
}

/* ================= EFFECTIVE ACTUATOR HELPERS ================= */
function getEffectiveManualLaser() {
  return state.latchLaser || state.holdLaserBtn || state.mouseLaserHeld;
}

function getEffectiveManualLed() {
  return state.latchLed || state.holdLedBtn || state.mouseLedHeld;
}

function pushUnifiedManualState({ includeAngles = !state.autoAim } = {}) {
  state.lastManualInputAt = performance.now();
  const patch = {
    led: getEffectiveManualLed(),
    laser: getEffectiveManualLaser(),
    includeAngles,
  };
  if (includeAngles) {
    patch.x = state.manualX;
    patch.y = state.manualY;
  }
  sf.update(patch);
  renderManualVisuals();
}

/* ================= MODE & TAB CONTROLS ================= */
function switchTab(tabId, btnEl) {
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
  const panel = document.getElementById(tabId);
  if (panel) panel.classList.add('active');
  if (btnEl) btnEl.classList.add('active');
}

async function setOperatingMode(autoAimEnabled, { silent = false } = {}) {
  const next = Boolean(autoAimEnabled);
  const changed = state.autoAim !== next;
  state.autoAim = next;

  if (next && state.pointerLocked && document.exitPointerLock) {
    document.exitPointerLock();
  }

  sf.sendAngles = !next;
  renderModeButtons();

  if (changed) {
    triggerHaptic(8);
    await sendConfig({ auto_aim: next });
    if (!silent) {
      showToast(next ? 'Auto-Aim (Visão Computacional) ativado' : 'Modo Manual ativado — Auto-Aim desligado');
    }
  }
}

function toggleOperatingModeFromGame() {
  setOperatingMode(!state.autoAim);
}

function renderModeButtons() {
  const autoBtn = document.getElementById('modeAutoBtn');
  const manBtn = document.getElementById('modeManualBtn');
  if (autoBtn) autoBtn.classList.toggle('active', state.autoAim);
  if (manBtn) manBtn.classList.toggle('active', !state.autoAim);

  const gameModeBtn = document.getElementById('gameModeSwitch');
  const gameModeText = document.getElementById('gameModeText');
  if (gameModeBtn) gameModeBtn.classList.toggle('manual', !state.autoAim);
  if (gameModeText) gameModeText.textContent = state.autoAim ? 'AUTO-AIM' : 'MANUAL';
}

/* ================= POINTER LOCK (DESKTOP FPS) ================= */
function enterPointerLock() {
  if (state.gameMode) {
    exitGameMode();
  }
  const stage = document.getElementById('videoStage');
  if (!stage || !stage.requestPointerLock) {
    enterGameMode();
    return;
  }

  if (state.autoAim) {
    setOperatingMode(false, { silent: true });
  }

  try {
    const req = stage.requestPointerLock();
    if (req && typeof req.catch === 'function') {
      req.catch(() => {
        showToast('Clique novamente no vídeo para travar o mouse.');
      });
    }
  } catch (_err) {
    enterGameMode();
  }
}

function onPointerLockChange() {
  const stage = document.getElementById('videoStage');
  const locked = document.pointerLockElement === stage;
  state.pointerLocked = locked;

  if (stage) stage.classList.toggle('pointer-locked', locked);
  const btn = document.getElementById('btnPointerLock');
  if (btn) {
    btn.querySelector('span').textContent = locked ? 'Mouse Preso (ESC sai)' : 'Travar Mouse (FPS)';
  }

  if (locked) {
    if (state.autoAim) {
      setOperatingMode(false, { silent: true });
    }
    sf.sendAngles = true;
    showToast('Mouse preso — Modo FPS ativo (ESC para liberar)', 2400);
  } else {
    let needRelease = false;
    if (state.mouseLaserHeld) {
      state.mouseLaserHeld = false;
      needRelease = true;
    }
    if (state.mouseLedHeld) {
      state.mouseLedHeld = false;
      needRelease = true;
    }
    if (needRelease) {
      pushUnifiedManualState({ includeAngles: !state.autoAim });
    }
  }

  renderManualVisuals();
}

function onPointerLockMouseMove(e) {
  if (!state.pointerLocked) return;

  const dxPx = Math.max(-220, Math.min(220, e.movementX || 0));
  const dyPx = Math.max(-220, Math.min(220, e.movementY || 0));
  if (dxPx === 0 && dyPx === 0) return;

  const dirX = state.invertX ? 1.0 : -1.0;
  const dirY = state.invertY ? -1.0 : 1.0;

  state.manualX = Math.max(0, Math.min(180, state.manualX + dirX * dxPx * state.mouseSensitivity));
  state.manualY = Math.max(0, Math.min(180, state.manualY + dirY * dyPx * state.mouseSensitivity));

  pushUnifiedManualState({ includeAngles: true });
}

function onPointerLockMouseDown(e) {
  if (!state.pointerLocked) return;
  e.preventDefault();
  if (e.button === 0) {
    state.mouseLaserHeld = true;
    pushUnifiedManualState({ includeAngles: true });
  } else if (e.button === 2) {
    state.mouseLedHeld = true;
    pushUnifiedManualState({ includeAngles: true });
  } else if (e.button === 1) {
    centerTurret();
  }
}

function onPointerLockMouseUp(e) {
  if (!state.pointerLocked) return;
  if (e.button === 0 && state.mouseLaserHeld) {
    state.mouseLaserHeld = false;
    pushUnifiedManualState({ includeAngles: true });
  } else if (e.button === 2 && state.mouseLedHeld) {
    state.mouseLedHeld = false;
    pushUnifiedManualState({ includeAngles: true });
  }
}

function onPointerLockWheel(e) {
  if (!state.pointerLocked) return;
  e.preventDefault();
  const delta = e.deltaY < 0 ? 0.02 : -0.02;
  state.mouseSensitivity = Math.max(0.05, Math.min(0.60, +(state.mouseSensitivity + delta).toFixed(2)));
  localStorage.setItem('autoaim_mouse_sens', String(state.mouseSensitivity));
  const msEl = document.getElementById('mouseSens');
  if (msEl) msEl.value = state.mouseSensitivity;
  renderManualVisuals();
}

/* ================= FULLSCREEN MOBILE GAME COCKPIT ================= */
async function enterGameMode() {
  if (state.pointerLocked && document.exitPointerLock) {
    document.exitPointerLock();
  }
  const stage = document.getElementById('videoStage');
  if (!stage) return;

  state.savedScrollY = typeof window.scrollY === 'number' ? window.scrollY : 0;
  state.gameMode = true;
  state.enteredNativeFs = false;
  stage.classList.add('game-mode');
  document.body.classList.add('game-mode-active');
  syncDrawerInputs();
  triggerHaptic(14);

  const fsTarget = document.documentElement || stage;
  const reqFs = fsTarget.requestFullscreen || fsTarget.webkitRequestFullscreen;
  if (typeof reqFs === 'function') {
    try {
      const p = reqFs.call(fsTarget);
      if (p && typeof p.then === 'function') {
        await p;
      }
      state.enteredNativeFs = true;
      if (screen.orientation && typeof screen.orientation.lock === 'function') {
        screen.orientation.lock('landscape').catch(() => {});
      }
    } catch (_err) {
      // Fixed-viewport CSS fallback still gives full-screen mobile cockpit (e.g. iOS Safari)
      state.enteredNativeFs = false;
    }
  }

  renderModeButtons();
  renderManualVisuals();
}

function exitGameMode({ skipExitFs = false } = {}) {
  if (!state.gameMode) return;
  state.gameMode = false;
  state.enteredNativeFs = false;
  toggleGameDrawer(false);
  releaseAllMomentaryInputs();

  const stage = document.getElementById('videoStage');
  if (stage) stage.classList.remove('game-mode');
  document.body.classList.remove('game-mode-active');

  if (!skipExitFs) {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    if (fsEl) {
      const exitFs = document.exitFullscreen || document.webkitExitFullscreen;
      if (typeof exitFs === 'function') {
        try {
          const p = exitFs.call(document);
          if (p && typeof p.catch === 'function') p.catch(() => {});
        } catch (_err) {}
      }
    }
    if (screen.orientation && typeof screen.orientation.unlock === 'function') {
      try { screen.orientation.unlock(); } catch (_err) {}
    }
  }

  if (typeof window.scrollTo === 'function' && state.savedScrollY > 0) {
    window.scrollTo(0, state.savedScrollY);
  }

  renderManualVisuals();
}

function toggleGameDrawer(forceState = null) {
  const drawer = document.getElementById('gameDrawer');
  if (!drawer) return;
  const next = forceState === null ? !state.gameDrawerOpen : Boolean(forceState);
  state.gameDrawerOpen = next;
  drawer.classList.toggle('open', next);
  drawer.setAttribute('aria-hidden', String(!next));
  if (next) syncDrawerInputs();
}

function syncDrawerInputs() {
  const gJoy = document.getElementById('gameJoySens');
  if (gJoy) gJoy.value = state.joystickSpeedDegPerSec;
  const gTouch = document.getElementById('gameTouchSens');
  if (gTouch) gTouch.value = state.touchAimSensitivity;
  const gEco = document.getElementById('gameEcoMode');
  if (gEco) gEco.checked = state.ecoMode;
  const gEcoFlash = document.getElementById('gameEcoMotionFlash');
  if (gEcoFlash) gEcoFlash.checked = state.ecoMotionFlash;
  const gIx = document.getElementById('gameInvX');
  if (gIx) gIx.checked = state.invertX;
  const gIy = document.getElementById('gameInvY');
  if (gIy) gIy.checked = state.invertY;
  updateLabels();
}

/* ================= VIRTUAL ANALOG JOYSTICK (60 FPS PROPORTIONAL) ================= */
function stopJoystick() {
  if (joystickState.rafId !== null) {
    cancelAnimationFrame(joystickState.rafId);
    joystickState.rafId = null;
  }
  if (joystickState.padEl) {
    joystickState.padEl.classList.remove('active');
  }
  if (joystickState.knobEl) {
    joystickState.knobEl.style.transform = 'translate3d(0px, 0px, 0px)';
  }
  joystickState.active = false;
  joystickState.pointerId = null;
  joystickState.padEl = null;
  joystickState.knobEl = null;
  joystickState.normX = 0;
  joystickState.normY = 0;
}

function tickJoystick(ts) {
  if (!joystickState.active) return;

  const dt = joystickState.lastFrameTs
    ? Math.min(0.05, Math.max(0.001, (ts - joystickState.lastFrameTs) / 1000.0))
    : 0.016;
  joystickState.lastFrameTs = ts;

  const nx = joystickState.normX;
  const ny = joystickState.normY;
  const mag = Math.hypot(nx, ny);
  const deadzone = 0.10;

  if (mag > deadzone) {
    const normR = Math.min(1.0, (mag - deadzone) / (1.0 - deadzone));
    // Smooth non-linear curve: fine precision near center, fast slew at rim
    const curved = normR * (0.35 + 0.65 * normR);
    const scale = curved / mag;
    const vx = nx * scale;
    const vy = ny * scale;

    const dirX = state.invertX ? 1.0 : -1.0;
    const dirY = state.invertY ? -1.0 : 1.0;

    const nextX = Math.max(0, Math.min(180, state.manualX + dirX * vx * state.joystickSpeedDegPerSec * dt));
    const nextY = Math.max(0, Math.min(180, state.manualY + dirY * vy * state.joystickSpeedDegPerSec * dt));

    if (nextX !== state.manualX || nextY !== state.manualY) {
      state.manualX = nextX;
      state.manualY = nextY;
      pushUnifiedManualState({ includeAngles: true });
    } else {
      state.lastManualInputAt = performance.now();
    }
  }

  joystickState.rafId = requestAnimationFrame(tickJoystick);
}

function bindJoystick(padId, knobId) {
  const pad = document.getElementById(padId);
  const knob = document.getElementById(knobId);
  if (!pad || !knob) return;

  const updateStickVector = (ev) => {
    const rect = pad.getBoundingClientRect();
    const cx = rect.left + rect.width * 0.5;
    const cy = rect.top + rect.height * 0.5;
    const knobRadius = (knob.offsetWidth || 44) * 0.5;
    const maxTravel = Math.max(20, rect.width * 0.5 - knobRadius * 0.55);

    let dx = ev.clientX - cx;
    let dy = ev.clientY - cy;
    const dist = Math.hypot(dx, dy);
    if (dist > maxTravel) {
      const k = maxTravel / dist;
      dx *= k;
      dy *= k;
    }

    joystickState.normX = dx / maxTravel;
    joystickState.normY = dy / maxTravel;
    knob.style.transform = `translate3d(${dx.toFixed(1)}px, ${dy.toFixed(1)}px, 0px)`;
  };

  pad.addEventListener('pointerdown', (ev) => {
    if (joystickState.active) return;
    ev.preventDefault();
    ev.stopPropagation();

    try {
      pad.setPointerCapture(ev.pointerId);
    } catch (_err) {}

    if (state.autoAim) {
      setOperatingMode(false, { silent: true });
    }

    joystickState.active = true;
    joystickState.pointerId = ev.pointerId;
    joystickState.padEl = pad;
    joystickState.knobEl = knob;
    joystickState.lastFrameTs = 0;

    pad.classList.add('active');
    triggerHaptic(8);
    updateStickVector(ev);

    if (joystickState.rafId !== null) {
      cancelAnimationFrame(joystickState.rafId);
    }
    joystickState.rafId = requestAnimationFrame(tickJoystick);
  });

  pad.addEventListener('pointermove', (ev) => {
    if (!joystickState.active || ev.pointerId !== joystickState.pointerId) return;
    ev.preventDefault();
    updateStickVector(ev);
  });

  const endStick = (ev) => {
    if (!joystickState.active || ev.pointerId !== joystickState.pointerId) return;
    stopJoystick();
  };

  pad.addEventListener('pointerup', endStick);
  pad.addEventListener('pointercancel', endStick);
  pad.addEventListener('lostpointercapture', endStick);
}

/* ================= DIRECT SWIPE-TO-AIM ON VIDEO (GAME MODE) ================= */
function initGameAimSurface() {
  const surface = document.getElementById('gameAimSurface');
  if (!surface) return;

  surface.addEventListener('pointerdown', (ev) => {
    if (!state.gameMode || swipeAimState.active) return;
    if (state.gameDrawerOpen) {
      toggleGameDrawer(false);
    }
    ev.preventDefault();
    try {
      surface.setPointerCapture(ev.pointerId);
    } catch (_err) {}

    swipeAimState.active = true;
    swipeAimState.pointerId = ev.pointerId;
    swipeAimState.lastX = ev.clientX;
    swipeAimState.lastY = ev.clientY;

    const hint = document.getElementById('gameAimHint');
    if (hint) hint.classList.add('faded');
  });

  surface.addEventListener('pointermove', (ev) => {
    if (!swipeAimState.active || ev.pointerId !== swipeAimState.pointerId) return;
    ev.preventDefault();

    const dxPx = Math.max(-160, Math.min(160, ev.clientX - swipeAimState.lastX));
    const dyPx = Math.max(-160, Math.min(160, ev.clientY - swipeAimState.lastY));
    swipeAimState.lastX = ev.clientX;
    swipeAimState.lastY = ev.clientY;

    if (dxPx === 0 && dyPx === 0) return;

    if (state.autoAim) {
      setOperatingMode(false, { silent: true });
    }

    const dirX = state.invertX ? 1.0 : -1.0;
    const dirY = state.invertY ? -1.0 : 1.0;

    state.manualX = Math.max(0, Math.min(180, state.manualX + dirX * dxPx * state.touchAimSensitivity));
    state.manualY = Math.max(0, Math.min(180, state.manualY + dirY * dyPx * state.touchAimSensitivity));

    pushUnifiedManualState({ includeAngles: true });
  });

  const endSwipe = (ev) => {
    if (!swipeAimState.active || ev.pointerId !== swipeAimState.pointerId) return;
    swipeAimState.active = false;
    swipeAimState.pointerId = null;
  };

  surface.addEventListener('pointerup', endSwipe);
  surface.addEventListener('pointercancel', endSwipe);
  surface.addEventListener('lostpointercapture', endSwipe);
}

/* ================= MULTI-TOUCH HOLD BUTTONS & ACTUATORS ================= */
function bindHoldActuatorButton(btnId, type) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  const ptrSet = type === 'laser' ? activeLaserPointers : activeLedPointers;

  const syncHold = () => {
    const active = ptrSet.size > 0;
    if (type === 'laser') {
      if (state.holdLaserBtn !== active) {
        state.holdLaserBtn = active;
        pushUnifiedManualState({ includeAngles: !state.autoAim });
      }
    } else if (state.holdLedBtn !== active) {
      state.holdLedBtn = active;
      pushUnifiedManualState({ includeAngles: !state.autoAim });
    }
    btn.classList.toggle('holding', active);
    btn.classList.toggle('firing', active);
  };

  btn.addEventListener('pointerdown', (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    try {
      btn.setPointerCapture(ev.pointerId);
    } catch (_err) {}
    ptrSet.add(ev.pointerId);
    triggerHaptic(12);
    syncHold();
  });

  const release = (ev) => {
    if (!ptrSet.has(ev.pointerId)) return;
    ptrSet.delete(ev.pointerId);
    syncHold();
  };

  btn.addEventListener('pointerup', release);
  btn.addEventListener('pointercancel', release);
  btn.addEventListener('lostpointercapture', release);
}

function releaseAllMomentaryInputs() {
  stopJoystick();
  swipeAimState.active = false;
  swipeAimState.pointerId = null;

  let changed = false;
  if (activeLaserPointers.size > 0 || state.holdLaserBtn || state.mouseLaserHeld) {
    activeLaserPointers.clear();
    state.holdLaserBtn = false;
    state.mouseLaserHeld = false;
    changed = true;
  }
  if (activeLedPointers.size > 0 || state.holdLedBtn || state.mouseLedHeld) {
    activeLedPointers.clear();
    state.holdLedBtn = false;
    state.mouseLedHeld = false;
    changed = true;
  }
  document.querySelectorAll('.fire-trigger, .pulse-btn').forEach((el) => {
    el.classList.remove('holding', 'firing');
  });
  if (changed) {
    pushUnifiedManualState({ includeAngles: !state.autoAim });
  }
}

function toggleManualLaser() {
  state.latchLaser = !state.latchLaser;
  const tl = document.getElementById('toggleLaser');
  if (tl) tl.checked = state.latchLaser;
  triggerHaptic(10);
  pushUnifiedManualState({ includeAngles: !state.autoAim });
}

function onLaserSwitchChange() {
  state.latchLaser = document.getElementById('toggleLaser').checked;
  pushUnifiedManualState({ includeAngles: !state.autoAim });
}

function toggleManualLed() {
  state.latchLed = !state.latchLed;
  const tl = document.getElementById('toggleLed');
  if (tl) tl.checked = state.latchLed;
  triggerHaptic(10);
  pushUnifiedManualState({ includeAngles: !state.autoAim });
}

function onLedSwitchChange() {
  state.latchLed = document.getElementById('toggleLed').checked;
  pushUnifiedManualState({ includeAngles: !state.autoAim });
}

function centerTurret() {
  state.manualX = 90.0;
  state.manualY = 90.0;
  triggerHaptic(8);
  if (!state.autoAim) {
    // Force transmission of 90,90 even if local cache already holds 90,90
    sf.sent.x = null;
    sf.sent.y = null;
    pushUnifiedManualState({ includeAngles: true });
  } else {
    fetch('/api/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'center' }),
    })
      .then((resp) => (resp.ok ? resp.json().catch(() => null) : null))
      .then((data) => {
        if (data && data.status) applyStatus(data.status);
      })
      .catch(() => {});
    renderManualVisuals();
  }
  showToast('Torreta centralizada (90°, 90°)', 1500);
}

/* ================= MANUAL SLIDERS, SENSITIVITY & 2D GIMBAL PAD ================= */
function onManualSliderInput() {
  if (state.autoAim) {
    setOperatingMode(false, { silent: true });
  }
  state.manualX = parseFloat(document.getElementById('sliderX').value);
  state.manualY = parseFloat(document.getElementById('sliderY').value);
  pushUnifiedManualState({ includeAngles: true });
}

function onMouseSensInput() {
  state.mouseSensitivity = parseFloat(document.getElementById('mouseSens').value);
  localStorage.setItem('autoaim_mouse_sens', String(state.mouseSensitivity));
  renderManualVisuals();
}

function onJoystickSensInput(val) {
  state.joystickSpeedDegPerSec = Math.max(30, Math.min(240, parseFloat(val) || 115));
  localStorage.setItem('autoaim_joy_sens', String(state.joystickSpeedDegPerSec));
  const s1 = document.getElementById('joySens');
  const s2 = document.getElementById('gameJoySens');
  if (s1) s1.value = state.joystickSpeedDegPerSec;
  if (s2) s2.value = state.joystickSpeedDegPerSec;
  updateLabels();
}

function onTouchSensInput(val) {
  state.touchAimSensitivity = Math.max(0.08, Math.min(0.60, parseFloat(val) || 0.24));
  localStorage.setItem('autoaim_touch_sens', String(state.touchAimSensitivity));
  const s1 = document.getElementById('touchSens');
  const s2 = document.getElementById('gameTouchSens');
  if (s1) s1.value = state.touchAimSensitivity;
  if (s2) s2.value = state.touchAimSensitivity;
  updateLabels();
}

function initGimbalPad() {
  const pad = document.getElementById('gimbalPad');
  if (!pad) return;
  let dragging = false;
  let activePid = null;

  const updateFromPointer = (ev) => {
    const rect = pad.getBoundingClientRect();
    const nx = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const ny = Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height));

    const angleX = state.invertX ? nx * 180 : (1 - nx) * 180;
    const angleY = state.invertY ? (1 - ny) * 180 : ny * 180;

    if (state.autoAim) {
      setOperatingMode(false, { silent: true });
    }
    state.manualX = angleX;
    state.manualY = angleY;
    pushUnifiedManualState({ includeAngles: true });
  };

  pad.addEventListener('pointerdown', (ev) => {
    ev.preventDefault();
    dragging = true;
    activePid = ev.pointerId;
    pad.classList.add('dragging');
    try { pad.setPointerCapture(ev.pointerId); } catch (_err) {}
    updateFromPointer(ev);
  });
  pad.addEventListener('pointermove', (ev) => {
    if (!dragging || ev.pointerId !== activePid) return;
    ev.preventDefault();
    updateFromPointer(ev);
  });
  const endDrag = (ev) => {
    if (ev.pointerId !== activePid) return;
    dragging = false;
    activePid = null;
    pad.classList.remove('dragging');
  };
  pad.addEventListener('pointerup', endDrag);
  pad.addEventListener('pointercancel', endDrag);
  pad.addEventListener('lostpointercapture', endDrag);
}

/* ================= CONFIG & ECO MODE CONTROLS ================= */
function updateLabels() {
  const kpEl = document.getElementById('kp');
  const dzEl = document.getElementById('deadzone');
  const cfEl = document.getElementById('conf');
  if (kpEl) document.getElementById('kpVal').textContent = parseFloat(kpEl.value).toFixed(2);
  if (dzEl) document.getElementById('deadzoneVal').textContent = `${dzEl.value} px`;
  if (cfEl) document.getElementById('confVal').textContent = parseFloat(cfEl.value).toFixed(2);

  const edEl = document.getElementById('ecoDelay');
  const efEl = document.getElementById('ecoFps');
  const emSensEl = document.getElementById('ecoMotionSens');
  const efDurEl = document.getElementById('ecoFlashDur');
  const delayVal = edEl ? parseInt(edEl.value, 10) : state.ecoDelayS;
  const fpsVal = efEl ? parseFloat(efEl.value) : state.ecoFps;
  const mSensVal = emSensEl ? parseInt(emSensEl.value, 10) : state.ecoMotionSensitivity;
  const fDurVal = efDurEl ? parseInt(efDurEl.value, 10) : state.ecoFlashDurationS;

  if (edEl) document.getElementById('ecoDelayVal').textContent = `${delayVal} s`;
  if (efEl) document.getElementById('ecoFpsVal').textContent = `${fpsVal.toFixed(1)} FPS`;
  if (emSensEl) document.getElementById('ecoMotionSensVal').textContent = `${mSensVal}%`;
  if (efDurEl) document.getElementById('ecoFlashDurVal').textContent = `${fDurVal} s`;

  const gEcoSub = document.getElementById('gameEcoSub');
  if (gEcoSub) gEcoSub.textContent = `${fpsVal.toFixed(1)} FPS após ${delayVal}s ocioso`;

  const jVal = document.getElementById('joySensVal');
  if (jVal) jVal.textContent = `${Math.round(state.joystickSpeedDegPerSec)} °/s`;
  const gjVal = document.getElementById('gameJoySensVal');
  if (gjVal) gjVal.textContent = `${Math.round(state.joystickSpeedDegPerSec)} °/s`;

  const tVal = document.getElementById('touchSensVal');
  if (tVal) tVal.textContent = `${state.touchAimSensitivity.toFixed(2)} °/px`;
  const gtVal = document.getElementById('gameTouchSensVal');
  if (gtVal) gtVal.textContent = `${state.touchAimSensitivity.toFixed(2)} °/px`;

  document.querySelectorAll('#ecoDelayPresets .preset-chip').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.delay, 10) === delayVal);
  });
  document.querySelectorAll('#ecoFpsPresets .preset-chip').forEach((btn) => {
    btn.classList.toggle('active', Math.abs(parseFloat(btn.dataset.fps) - fpsVal) < 0.05);
  });

  const dcEl = document.getElementById('discordCooldown');
  const daEl = document.getElementById('discordAbsence');
  const cdVal = dcEl ? parseInt(dcEl.value, 10) : state.discordCooldownS;
  const abVal = daEl ? parseInt(daEl.value, 10) : state.discordAbsenceS;
  if (dcEl) document.getElementById('discordCooldownVal').textContent = `${cdVal} s`;
  if (daEl) document.getElementById('discordAbsenceVal').textContent = `${abVal} s`;
  document.querySelectorAll('#discordCooldownPresets .preset-chip').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.cd, 10) === cdVal);
  });

  const briEl = document.getElementById('camBri');
  const aeEl = document.getElementById('camAe');
  const gainEl = document.getElementById('camGain');
  const briVal = briEl ? parseInt(briEl.value, 10) : state.camBri;
  const aeVal = aeEl ? parseInt(aeEl.value, 10) : state.camAe;
  const gainVal = gainEl ? parseInt(gainEl.value, 10) : state.camGain;
  if (briEl) {
    document.getElementById('camBriVal').textContent = briVal > 0 ? `+${briVal}` : String(briVal);
  }
  if (aeEl) {
    document.getElementById('camAeVal').textContent = aeVal > 0 ? `+${aeVal}` : String(aeVal);
  }
  if (gainEl) {
    document.getElementById('camGainVal').textContent = CAM_GAIN_NAMES[gainVal] || `${gainVal}`;
  }
  document.querySelectorAll('#camGainPresets .preset-chip').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.gain, 10) === gainVal);
  });
}

function onDiscordConfigChange() {
  state.discordEnabled = document.getElementById('discordEnabled').checked;
  state.discordCooldownS = parseInt(document.getElementById('discordCooldown').value, 10);
  state.discordAbsenceS = parseInt(document.getElementById('discordAbsence').value, 10);
  state.discordIncludeScene = document.getElementById('discordIncludeScene').checked;
  state.discordLogDogs = document.getElementById('discordLogDogs').checked;
  updateLabels();
  sendConfig({
    discord_enabled: state.discordEnabled,
    discord_cooldown_s: state.discordCooldownS,
    discord_absence_reset_s: state.discordAbsenceS,
    discord_include_scene: state.discordIncludeScene,
    discord_log_dogs: state.discordLogDogs,
  });
}

function setDiscordCooldownPreset(seconds) {
  const el = document.getElementById('discordCooldown');
  if (el) el.value = String(seconds);
  onDiscordConfigChange();
}

async function testDiscordWebhook() {
  const btn = document.getElementById('btnDiscordTest');
  if (btn) btn.disabled = true;
  try {
    const resp = await fetch('/api/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'discord_test' }),
    });
    const data = await resp.json().catch(() => ({}));
    if (resp.ok && data && data.status) {
      applyStatus(data.status);
    }
    showToast(data.message || (resp.ok ? 'Teste enviado ao Discord' : 'Falha ao enviar teste'), 2500);
  } catch (_err) {
    showToast('Erro de rede ao testar webhook do Discord', 2500);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function onEcoConfigChange() {
  state.ecoMode = document.getElementById('ecoMode').checked;
  state.ecoDelayS = parseInt(document.getElementById('ecoDelay').value, 10);
  state.ecoFps = parseFloat(document.getElementById('ecoFps').value);
  state.ecoMotionWake = document.getElementById('ecoMotionWake').checked;
  state.ecoMotionFlash = document.getElementById('ecoMotionFlash').checked;
  state.ecoFlashDarkOnly = document.getElementById('ecoFlashDarkOnly').checked;
  state.ecoMotionSensitivity = parseInt(document.getElementById('ecoMotionSens').value, 10);
  state.ecoFlashDurationS = parseInt(document.getElementById('ecoFlashDur').value, 10);

  const gEco = document.getElementById('gameEcoMode');
  if (gEco) gEco.checked = state.ecoMode;
  const gEcoFlash = document.getElementById('gameEcoMotionFlash');
  if (gEcoFlash) gEcoFlash.checked = state.ecoMotionFlash;
  updateLabels();
  sendConfig({
    eco_mode: state.ecoMode,
    eco_delay_s: state.ecoDelayS,
    eco_fps: state.ecoFps,
    eco_motion_wake: state.ecoMotionWake,
    eco_motion_flash: state.ecoMotionFlash,
    eco_flash_dark_only: state.ecoFlashDarkOnly,
    eco_motion_sensitivity: state.ecoMotionSensitivity,
    eco_flash_duration_s: state.ecoFlashDurationS,
  });
}

function onGameEcoToggle(checked) {
  const el = document.getElementById('ecoMode');
  if (el) el.checked = Boolean(checked);
  onEcoConfigChange();
}

function onGameEcoFlashToggle(checked) {
  const el = document.getElementById('ecoMotionFlash');
  if (el) el.checked = Boolean(checked);
  onEcoConfigChange();
}

function setEcoDelayPreset(seconds) {
  const el = document.getElementById('ecoDelay');
  if (el) el.value = String(seconds);
  onEcoConfigChange();
}

function setEcoFpsPreset(fps) {
  const el = document.getElementById('ecoFps');
  if (el) el.value = String(fps);
  onEcoConfigChange();
}

function syncInvertFromDrawer() {
  const gIx = document.getElementById('gameInvX');
  const gIy = document.getElementById('gameInvY');
  if (gIx) document.getElementById('invX').checked = gIx.checked;
  if (gIy) document.getElementById('invY').checked = gIy.checked;
  sendConfig();
}

function onAutoFireChange() {
  const af = document.getElementById('autoFire').checked;
  const lm = document.getElementById('ledMode');
  if (af && lm.value === 'off') {
    lm.value = 'on_lock';
  }
  sendConfig();
}

function onLedModeChange() {
  const lm = document.getElementById('ledMode').value;
  document.getElementById('autoFire').checked = lm !== 'off';
  sendConfig();
}

function setAutoFireOutput(val) {
  state.autoFireLaser = val === 'laser' || val === 'both';
  state.autoFireLed = val === 'led' || val === 'both';
  renderSegmented('#fireOutputSeg', 'val', val);
  sendConfig({
    auto_fire_laser: state.autoFireLaser,
    auto_fire_led: state.autoFireLed,
  });
}

function setAimMode(val) {
  state.aimMode = val;
  renderSegmented('#aimModeSeg', 'val', val);
  renderSegmented('#gameAimModeSeg', 'val', val);
  sendConfig({ aim_mode: val });
}

function setRotation(rot) {
  state.rotation = rot;
  renderSegmented('#rotationSeg', 'rot', String(rot));
  sendConfig({ rotation: rot });
}

async function sendCamCommand(patch = {}) {
  const params = new URLSearchParams();
  if (patch.res !== undefined && patch.res !== null) {
    params.set('res', String(Math.max(0, Math.min(3, parseInt(patch.res, 10)))));
  }
  if (patch.bri !== undefined && patch.bri !== null) {
    params.set('bri', String(Math.max(-3, Math.min(3, parseInt(patch.bri, 10)))));
  }
  if (patch.ae !== undefined && patch.ae !== null) {
    params.set('ae', String(Math.max(-5, Math.min(5, parseInt(patch.ae, 10)))));
  }
  if (patch.gain !== undefined && patch.gain !== null) {
    params.set('gain', String(Math.max(0, Math.min(5, parseInt(patch.gain, 10)))));
  }
  if (patch.night !== undefined && patch.night !== null) {
    params.set('night', patch.night ? '1' : '0');
  }
  const qs = params.toString();
  if (!qs) return;

  state.lastCamSentAt = performance.now();
  const url = `/cam?${qs}`;
  const elProto = document.getElementById('protoPreview');
  if (elProto) elProto.textContent = `GET ${url}`;

  try {
    await fetch(url, { method: 'GET', cache: 'no-store' });
  } catch (_err) {
    // ignore transient network errors
  }
}

function selectResolution(code) {
  const clamped = Math.max(0, Math.min(3, parseInt(code, 10)));
  state.resolution = clamped;
  renderResolutionGrid(clamped);
  sendCamCommand({ res: clamped });
  showToast(`Resolução (/cam): ${RES_NAMES[clamped] || clamped}`, 1800);
}

function onCamBriChange() {
  const val = Math.max(-3, Math.min(3, parseInt(document.getElementById('camBri').value, 10)));
  state.camBri = val;
  updateLabels();
  sendCamCommand({ bri: val });
}

function onCamAeChange() {
  const val = Math.max(-5, Math.min(5, parseInt(document.getElementById('camAe').value, 10)));
  state.camAe = val;
  updateLabels();
  sendCamCommand({ ae: val });
}

function onCamGainChange() {
  const val = Math.max(0, Math.min(5, parseInt(document.getElementById('camGain').value, 10)));
  state.camGain = val;
  updateLabels();
  sendCamCommand({ gain: val });
}

function setCamGainPreset(gainVal) {
  const el = document.getElementById('camGain');
  if (el) el.value = String(gainVal);
  onCamGainChange();
}

function onCamNightChange() {
  const val = Boolean(document.getElementById('camNight').checked);
  state.camNight = val;
  sendCamCommand({ night: val });
}

function resetCamDefaults() {
  state.camBri = 1;
  state.camAe = 3;
  state.camGain = 5;
  state.camNight = true;
  const elBri = document.getElementById('camBri');
  const elAe = document.getElementById('camAe');
  const elGain = document.getElementById('camGain');
  const elNight = document.getElementById('camNight');
  if (elBri) elBri.value = '1';
  if (elAe) elAe.value = '3';
  if (elGain) elGain.value = '5';
  if (elNight) elNight.checked = true;
  updateLabels();
  sendCamCommand({ bri: 1, ae: 3, gain: 5, night: true });
  showToast('Sensor OV3660 restaurado (bri=+1, ae=+3, 64x, night=1)', 2000);
}

function renderSegmented(selector, attr, activeValue) {
  document.querySelectorAll(`${selector} .seg-btn`).forEach((btn) => {
    btn.classList.toggle('active', btn.dataset[attr] === String(activeValue));
  });
}

function renderResolutionGrid(activeCode) {
  document.querySelectorAll('#resGrid .res-btn, #gameResGrid .res-btn').forEach((btn) => {
    btn.classList.toggle('active', parseInt(btn.dataset.res, 10) === activeCode);
  });
}

async function sendConfig(extra = {}) {
  state.lastConfigSentAt = performance.now();
  updateLabels();
  state.invertX = document.getElementById('invX').checked;
  state.invertY = document.getElementById('invY').checked;
  state.ecoMode = document.getElementById('ecoMode').checked;
  state.ecoDelayS = parseInt(document.getElementById('ecoDelay').value, 10);
  state.ecoFps = parseFloat(document.getElementById('ecoFps').value);
  state.ecoMotionWake = document.getElementById('ecoMotionWake').checked;
  state.ecoMotionFlash = document.getElementById('ecoMotionFlash').checked;
  state.ecoFlashDarkOnly = document.getElementById('ecoFlashDarkOnly').checked;
  state.ecoMotionSensitivity = parseInt(document.getElementById('ecoMotionSens').value, 10);
  state.ecoFlashDurationS = parseInt(document.getElementById('ecoFlashDur').value, 10);
  state.discordEnabled = document.getElementById('discordEnabled').checked;
  state.discordCooldownS = parseInt(document.getElementById('discordCooldown').value, 10);
  state.discordAbsenceS = parseInt(document.getElementById('discordAbsence').value, 10);
  state.discordIncludeScene = document.getElementById('discordIncludeScene').checked;
  state.discordLogDogs = document.getElementById('discordLogDogs').checked;

  const gIx = document.getElementById('gameInvX');
  const gIy = document.getElementById('gameInvY');
  if (gIx) gIx.checked = state.invertX;
  if (gIy) gIy.checked = state.invertY;

  const payload = {
    auto_aim: state.autoAim,
    detect_person: document.getElementById('detectPerson').checked,
    detect_dog: document.getElementById('detectDog').checked,
    detect_in_manual: document.getElementById('detectInManual').checked,
    eco_mode: state.ecoMode,
    eco_delay_s: state.ecoDelayS,
    eco_fps: state.ecoFps,
    eco_motion_wake: state.ecoMotionWake,
    eco_motion_flash: state.ecoMotionFlash,
    eco_flash_dark_only: state.ecoFlashDarkOnly,
    eco_motion_sensitivity: state.ecoMotionSensitivity,
    eco_flash_duration_s: state.ecoFlashDurationS,
    discord_enabled: state.discordEnabled,
    discord_cooldown_s: state.discordCooldownS,
    discord_absence_reset_s: state.discordAbsenceS,
    discord_include_scene: state.discordIncludeScene,
    discord_log_dogs: state.discordLogDogs,
    auto_fire: document.getElementById('autoFire').checked,
    auto_fire_laser: state.autoFireLaser,
    auto_fire_led: state.autoFireLed,
    aim_mode: state.aimMode,
    led_mode: document.getElementById('ledMode').value,
    kp: parseFloat(document.getElementById('kp').value),
    deadzone_px: parseFloat(document.getElementById('deadzone').value),
    conf_threshold: parseFloat(document.getElementById('conf').value),
    invert_x: state.invertX,
    invert_y: state.invertY,
    rotation: state.rotation,
    ...extra,
  };

  try {
    const resp = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (resp.ok) {
      const data = await resp.json().catch(() => null);
      if (data && data.status) {
        applyStatus(data.status);
      }
    }
  } catch (_err) {
    // ignore transient errors
  }
}

/* ================= RENDER LOCAL MANUAL CONTROLS & HUD ================= */
function renderManualVisuals(serverLaser = null, serverLed = null) {
  const rx = Math.round(state.manualX);
  const ry = Math.round(state.manualY);
  const laserActive = serverLaser !== null ? serverLaser : getEffectiveManualLaser();
  const ledActive = serverLed !== null ? serverLed : getEffectiveManualLed();

  const sx = document.getElementById('sliderX');
  const sy = document.getElementById('sliderY');
  if (sx) sx.value = rx;
  if (sy) sy.value = ry;
  const sxVal = document.getElementById('sliderXVal');
  const syVal = document.getElementById('sliderYVal');
  if (sxVal) sxVal.textContent = `${rx}°`;
  if (syVal) syVal.textContent = `${ry}°`;

  const msVal = document.getElementById('mouseSensVal');
  if (msVal) msVal.textContent = `${state.mouseSensitivity.toFixed(2)} °/px`;
  const gc = document.getElementById('gimbalCoords');
  if (gc) gc.textContent = `${rx}° · ${ry}°`;
  const ic = document.getElementById('inlineCoords');
  if (ic) ic.textContent = `${rx}° · ${ry}°`;

  const px = state.invertX ? (rx / 180) * 100 : (1 - rx / 180) * 100;
  const py = state.invertY ? (1 - ry / 180) * 100 : (ry / 180) * 100;
  const thumb = document.getElementById('gimbalThumb');
  if (thumb) {
    thumb.style.left = `${px}%`;
    thumb.style.top = `${py}%`;
  }
  const guideH = document.getElementById('gimbalGuideH');
  const guideV = document.getElementById('gimbalGuideV');
  if (guideH) guideH.style.top = `${py}%`;
  if (guideV) guideV.style.left = `${px}%`;

  const gTop = document.getElementById('gimbalTop');
  const gBottom = document.getElementById('gimbalBottom');
  const gLeft = document.getElementById('gimbalLeft');
  const gRight = document.getElementById('gimbalRight');
  if (gTop) gTop.textContent = state.invertY ? 'Y 180°' : 'Y 0°';
  if (gBottom) gBottom.textContent = state.invertY ? 'Y 0°' : 'Y 180°';
  if (gLeft) gLeft.textContent = state.invertX ? 'X 0°' : 'X 180°';
  if (gRight) gRight.textContent = state.invertX ? 'X 180°' : 'X 0°';

  // Desktop FPS HUD overlays
  const hudX = document.getElementById('hudX');
  const hudY = document.getElementById('hudY');
  const hudSens = document.getElementById('hudSens');
  if (hudX) hudX.textContent = `${rx}°`;
  if (hudY) hudY.textContent = `${ry}°`;
  if (hudSens) hudSens.textContent = `${state.mouseSensitivity.toFixed(2)}°/px`;

  const hudLaser = document.getElementById('hudLaserChip');
  if (hudLaser) {
    hudLaser.textContent = laserActive ? 'LASER ON' : 'LASER OFF';
    hudLaser.classList.toggle('laser-active', laserActive);
  }
  const hudLed = document.getElementById('hudLedChip');
  if (hudLed) {
    hudLed.textContent = ledActive ? 'FLASH ON' : 'FLASH OFF';
    hudLed.classList.toggle('flash-active', ledActive);
  }

  const reticle = document.getElementById('fpsReticle');
  if (reticle) {
    reticle.classList.toggle('laser-active', laserActive);
    reticle.classList.toggle('flash-active', ledActive);
  }

  // Mobile / Game Mode HUD overlays
  const gameHudX = document.getElementById('gameHudX');
  const gameHudY = document.getElementById('gameHudY');
  const stickLbl = document.getElementById('gameStickLabel');
  if (gameHudX) gameHudX.textContent = `${rx}°`;
  if (gameHudY) gameHudY.textContent = `${ry}°`;
  if (stickLbl) stickLbl.textContent = `JOYSTICK · ${rx}° / ${ry}°`;

  const gRet = document.getElementById('gameReticle');
  if (gRet) {
    gRet.classList.toggle('laser-active', laserActive);
    gRet.classList.toggle('flash-active', ledActive);
    gRet.classList.toggle('target-locked', state.targetLocked);
  }

  const stage = document.getElementById('videoStage');
  if (stage) stage.classList.toggle('laser-firing', laserActive);

  // Quick Dock, Mobile Arcade & Actuator Indicators
  document.getElementById('btnQuickLaser')?.classList.toggle('active', laserActive);
  document.getElementById('btnQuickLed')?.classList.toggle('active', ledActive);
  document.getElementById('inlineBtnLatchLaser')?.classList.toggle('active', state.latchLaser);
  document.getElementById('inlineBtnFlash')?.classList.toggle('active', ledActive);
  document.getElementById('gameBtnLatchLaser')?.classList.toggle('active', state.latchLaser);
  document.getElementById('gameBtnFlash')?.classList.toggle('active', ledActive);
  document.getElementById('gameBtnFireLaser')?.classList.toggle('firing', laserActive);
  document.getElementById('inlineBtnFireLaser')?.classList.toggle('firing', laserActive);

  document.getElementById('dotLaser')?.classList.toggle('on', laserActive);
  document.getElementById('dotLed')?.classList.toggle('on', ledActive);
  const tLaser = document.getElementById('toggleLaser');
  const tLed = document.getElementById('toggleLed');
  if (tLaser) tLaser.checked = state.latchLaser;
  if (tLed) tLed.checked = state.latchLed;

  if (!state.autoAim) {
    document.getElementById('mAngles').textContent = `${rx}° / ${ry}°`;
  }
}

/* ================= STATUS POLLING ================= */
const STATUS_POLL_ACTIVE_MS = 1000;
const STATUS_POLL_IDLE_MS = 2500;
const STATUS_POLL_MAX_BACKOFF_MS = 5000;
const STATUS_FETCH_TIMEOUT_MS = 4000;

const statusPollState = {
  timerId: null,
  inFlight: false,
  consecutiveErrors: 0,
};

function getNextStatusPollDelayMs() {
  if (statusPollState.consecutiveErrors > 0) {
    return Math.min(
      STATUS_POLL_MAX_BACKOFF_MS,
      2000 * Math.pow(2, statusPollState.consecutiveErrors - 1)
    );
  }
  if (!state.initialized) {
    return STATUS_POLL_ACTIVE_MS;
  }
  if (!state.autoAim || (state.ecoActive && !state.ecoMotionWoken)) {
    return STATUS_POLL_IDLE_MS;
  }
  return STATUS_POLL_ACTIVE_MS;
}

function stopStatusPolling() {
  if (statusPollState.timerId !== null) {
    clearTimeout(statusPollState.timerId);
    statusPollState.timerId = null;
  }
}

function scheduleNextStatusPoll(delayMs = null) {
  stopStatusPolling();
  if (document.hidden) return;
  const delay = delayMs !== null ? delayMs : getNextStatusPollDelayMs();
  statusPollState.timerId = setTimeout(pollStatus, delay);
}

function applyStatus(s) {
  if (!s || typeof s !== 'object') return;

  // Header Status Pills
  const pStream = document.getElementById('pillStream');
  pStream.className = 'status-pill ' + (s.stream_connected ? 'ok' : 'warn');
  pStream.querySelector('.pill-text').textContent = s.stream_connected ? 'STREAM ONLINE' : 'RECONECTANDO';

  const gStream = document.getElementById('gameStreamPill');
  if (gStream) {
    gStream.className = 'game-chip ' + (s.stream_connected ? 'ok' : 'warn');
    document.getElementById('gameStreamText').textContent = s.stream_connected ? 'ONLINE' : 'RECONECT.';
  }

  const pDevice = document.getElementById('pillDevice');
  if (s.device && pDevice) {
    pDevice.className = 'status-pill pill-device ' + (s.device.includes('GPU') ? 'ok' : '');
    pDevice.querySelector('.pill-text').textContent = `HW: ${s.device}`;
  }

  // Eco Mode Pill & Badge
  const cvRunning = Boolean(s.auto_aim || s.detect_in_manual) && Boolean(s.detect_person || s.detect_dog);
  const ecoEnabled = s.eco_mode !== undefined ? Boolean(s.eco_mode) : state.ecoMode;
  const ecoActive = Boolean(s.eco_active);
  const ecoMotionWoken = Boolean(s.eco_motion_woken);
  const ecoFlashActive = Boolean(s.eco_flash_active);
  const ecoDelay = Number(s.eco_delay_s ?? state.ecoDelayS);
  const ecoFps = Number(s.eco_fps ?? state.ecoFps);
  const ecoIdle = Number(s.eco_idle_s ?? 0);
  state.ecoActive = ecoActive;
  state.ecoMotionWoken = ecoMotionWoken;

    const lumaEl = document.getElementById('ecoLumaVal');
    if (lumaEl && typeof s.ambient_luma === 'number') {
      lumaEl.textContent = String(Math.round(s.ambient_luma));
    }

    const pEco = document.getElementById('pillEco');
    const ecoChip = document.getElementById('ecoStatusChip');
    const stageEco = document.getElementById('stageEcoBadge');

    if (!ecoEnabled) {
      if (pEco) {
        pEco.className = 'status-pill';
        pEco.querySelector('.pill-text').textContent = 'ECO OFF';
      }
      if (ecoChip) {
        ecoChip.textContent = 'DESLIGADO';
        ecoChip.classList.remove('active', 'motion');
      }
      if (stageEco) stageEco.classList.add('hidden');
    } else if (!cvRunning) {
      if (pEco) {
        pEco.className = 'status-pill eco-standby';
        pEco.querySelector('.pill-text').textContent = 'ECO PAUSA';
      }
      if (ecoChip) {
        ecoChip.textContent = 'IA PAUSADA';
        ecoChip.classList.remove('active', 'motion');
      }
      if (stageEco) stageEco.classList.add('hidden');
    } else if (ecoMotionWoken) {
      const tag = ecoFlashActive ? 'MOV · LUZ ON' : 'MOV · CHECK';
      if (pEco) {
        pEco.className = 'status-pill eco-motion';
        pEco.querySelector('.pill-text').textContent = tag;
      }
      if (ecoChip) {
        ecoChip.textContent = tag;
        ecoChip.classList.remove('active');
        ecoChip.classList.add('motion');
      }
      if (stageEco) {
        stageEco.textContent = tag;
        stageEco.classList.remove('hidden');
      }
    } else if (ecoActive) {
      const fpsLabel = `${ecoFps.toFixed(1)} FPS`;
      if (pEco) {
        pEco.className = 'status-pill eco-active';
        pEco.querySelector('.pill-text').textContent = `ECO · ${fpsLabel}`;
      }
      if (ecoChip) {
        ecoChip.textContent = `ATIVO · ${fpsLabel}`;
        ecoChip.classList.add('active');
        ecoChip.classList.remove('motion');
      }
      if (stageEco) {
        stageEco.textContent = `ECO · ${fpsLabel}`;
        stageEco.classList.remove('hidden');
      }
    } else {
      const rem = Math.max(0, Math.ceil(ecoDelay - ecoIdle));
      if (pEco) {
        pEco.className = 'status-pill eco-standby';
        pEco.querySelector('.pill-text').textContent = `ECO ${rem}s`;
      }
      if (ecoChip) {
        ecoChip.textContent = `EM ${rem}s`;
        ecoChip.classList.remove('active', 'motion');
      }
      if (stageEco) stageEco.classList.add('hidden');
    }

    // Avoid fighting user's live manual input
    const activelyTouching = joystickState.active || swipeAimState.active || state.holdLaserBtn || state.holdLedBtn;
    const recentlyControlled = performance.now() - state.lastManualInputAt < 550 || sf.hasPending() || state.pointerLocked || activelyTouching;

    const effectiveLaserForUi = (!state.autoAim && recentlyControlled)
      ? getEffectiveManualLaser()
      : Boolean(s.laser_on);
    const effectiveLedForUi = (!state.autoAim && recentlyControlled)
      ? getEffectiveManualLed()
      : Boolean(s.led_on);

    const pLaser = document.getElementById('pillLaser');
    pLaser.className = 'status-pill ' + (effectiveLaserForUi ? 'laser-on' : '');
    pLaser.querySelector('.pill-text').textContent = effectiveLaserForUi ? 'LASER ON' : 'LASER OFF';

    const pLed = document.getElementById('pillLed');
    pLed.className = 'status-pill ' + (effectiveLedForUi ? 'flash-on' : '');
    pLed.querySelector('.pill-text').textContent = effectiveLedForUi ? 'FLASH ON' : 'FLASH OFF';

    // Target badge & telemetry
    state.targetLocked = Boolean(s.person_detected);
    const tgtName = TARGET_NAMES[s.target_type] || 'ALVO';
    const stageBadge = document.getElementById('stageTargetBadge');
    const gameBadge = document.getElementById('gameTargetBadge');

    if (s.person_detected) {
      const label = `ALVO: ${tgtName} (${s.detections_count})`;
      stageBadge.textContent = label;
      stageBadge.className = 'stage-badge locked-target';
      if (gameBadge) {
        gameBadge.textContent = label;
        gameBadge.className = 'game-chip locked-target';
      }
      document.getElementById('mTargetSub').textContent = `${tgtName} (${s.detections_count})`;
    } else {
      stageBadge.textContent = ecoMotionWoken
        ? (ecoFlashActive ? 'MOVIMENTO · LUZ ON' : 'MOVIMENTO DETECTADO')
        : (s.auto_aim ? 'ALVO: BUSCANDO' : 'MODO MANUAL');
      stageBadge.className = 'stage-badge';
      if (gameBadge) {
        if (ecoMotionWoken) {
          gameBadge.textContent = ecoFlashActive ? 'MOV · LUZ' : 'MOVIMENTO';
          gameBadge.className = 'game-chip eco-on';
        } else if (ecoActive) {
          gameBadge.textContent = `ECO ${ecoFps.toFixed(1)} FPS`;
          gameBadge.className = 'game-chip eco-on';
        } else {
          gameBadge.textContent = s.auto_aim ? 'BUSCANDO' : 'MANUAL';
          gameBadge.className = 'game-chip';
        }
      }
      document.getElementById('mTargetSub').textContent = s.auto_aim
        ? (ecoMotionWoken ? 'Movimento · Verificando...' : (ecoActive ? `Eco (${ecoFps.toFixed(1)} FPS)` : 'Buscando alvo...'))
        : 'Controle manual';
    }

    const displayFps = typeof s.fps === 'number' ? s.fps : 0.0;
    document.getElementById('mError').textContent = `${s.error_px[0]}, ${s.error_px[1]} px`;
    document.getElementById('mPerf').textContent = `${displayFps.toFixed(1)} / ${Math.round(s.inference_ms)}ms`;
    const gFps = document.getElementById('gameHudFps');
    if (gFps) gFps.textContent = displayFps.toFixed(1);

    if (s.frame_size && s.frame_size[0] > 0) {
      const resShort = RES_NAMES[s.resolution]?.split(' ')[0] || '';
      const vidFpsInfo = ecoActive && typeof s.video_fps === 'number'
        ? `ECO · Vídeo ${s.video_fps.toFixed(0)} FPS`
        : `${s.frame_size[0]}×${s.frame_size[1]} · ${resShort}`;
      document.getElementById('mFrameRes').textContent = vidFpsInfo;
    }

    if (typeof s.hw_move_interval_ms === 'number' && s.hw_move_interval_ms >= 10) {
      state.hwMoveIntervalMs = s.hw_move_interval_ms;
      sf.moveIntervalMs = s.hw_move_interval_ms;
    }
    if (typeof s.hw_last_payload === 'string' && s.hw_last_payload) {
      state.hwLastPayload = s.hw_last_payload;
      const elProto = document.getElementById('protoPreview');
      if (elProto) elProto.textContent = s.hw_last_payload;
    }

    const hwInfo = document.getElementById('metaTransportInfo');
    const manualBadge = document.getElementById('manualStatusBadge');
    if (hwInfo) {
      hwInfo.textContent = `HTTP keep-alive (/servo + /cam) · ${state.hwMoveIntervalMs} ms · Auto-Detach 500ms`;
    }
    if (manualBadge) {
      manualBadge.textContent = `${state.hwMoveIntervalMs} ms HTTP · Single-Flight`;
    }

    if (sf.lastRttMs === null && typeof s.hw_rtt_ms === 'number') {
      document.getElementById('mRtt').textContent = `${Math.round(s.hw_rtt_ms)} ms`;
      const gRtt = document.getElementById('gameHudRtt');
      if (gRtt) gRtt.textContent = `${Math.round(s.hw_rtt_ms)}ms`;
      const elTag = document.getElementById('mPriorityTag');
      if (elTag) {
        elTag.textContent = `0ms Prio · ${state.hwMoveIntervalMs}ms Mov`;
      }
    }

    if (s.turret_url) {
      document.getElementById('metaTurretUrl').textContent = s.turret_url;
    }
    document.getElementById('metaHwError').textContent = s.hw_error ? `Erro HW: ${s.hw_error}` : '';

    // Discord Face Log Telemetry
    const dConfigured = Boolean(s.discord_configured);
    const dEnabled = Boolean(s.discord_enabled);
    const dSent = Number(s.discord_sent_count ?? 0);
    const dState = String(s.discord_state || 'idle');
    const dErr = String(s.discord_last_error || '');
    state.discordConfigured = dConfigured;

    const dChip = document.getElementById('discordStatusChip');
    const dMeta = document.getElementById('discordMetaText');
    const dTestBtn = document.getElementById('btnDiscordTest');
    if (dTestBtn) dTestBtn.disabled = !dConfigured;

    if (dChip) {
      dChip.title = dErr || '';
      if (!dConfigured) {
        dChip.textContent = 'SEM .ENV';
        dChip.className = 'chip-meta discord-chip warn';
      } else if (dErr) {
        if (/timeout/i.test(dErr)) {
          dChip.textContent = 'TIMEOUT';
        } else if (/conex|connect/i.test(dErr)) {
          dChip.textContent = 'SEM CONEXÃO';
        } else {
          dChip.textContent = dErr.slice(0, 16);
        }
        dChip.className = 'chip-meta discord-chip warn';
      } else if (!dEnabled) {
        dChip.textContent = 'PAUSADO';
        dChip.className = 'chip-meta discord-chip';
      } else if (dState === 'collecting') {
        dChip.textContent = 'CAPTURANDO';
        dChip.className = 'chip-meta discord-chip collecting';
      } else if (dState === 'logged') {
        dChip.textContent = 'VISITA OK';
        dChip.className = 'chip-meta discord-chip ok';
      } else {
        dChip.textContent = 'ATIVO';
        dChip.className = 'chip-meta discord-chip ok';
      }
    }
    if (dMeta) {
      if (!dConfigured) {
        dMeta.textContent = 'Defina DISCORD_WEBHOOK_URL no .env';
      } else {
        const lastStr = s.discord_last_sent_ts
          ? ` · Último ${new Date(Number(s.discord_last_sent_ts) * 1000).toLocaleTimeString()}`
          : '';
        dMeta.textContent = `Webhook .env · ${dSent} ${dSent === 1 ? 'envio' : 'envios'}${lastStr}`;
      }
    }

    if (!state.initialized) {
      state.autoAim = Boolean(s.auto_aim);
      state.aimMode = s.aim_mode || 'center';
      state.rotation = s.rotation ?? 90;
      state.resolution = Math.max(0, Math.min(3, s.resolution ?? 3));
      state.camBri = typeof s.cam_bri === 'number' ? s.cam_bri : 1;
      state.camAe = typeof s.cam_ae === 'number' ? s.cam_ae : 3;
      state.camGain = typeof s.cam_gain === 'number' ? s.cam_gain : 5;
      state.camNight = s.cam_night !== undefined ? Boolean(s.cam_night) : true;
      state.autoFireLaser = s.auto_fire_laser !== undefined ? Boolean(s.auto_fire_laser) : true;
      state.autoFireLed = s.auto_fire_led !== undefined ? Boolean(s.auto_fire_led) : false;
      state.invertX = Boolean(s.invert_x);
      state.invertY = Boolean(s.invert_y);
      state.ecoMode = s.eco_mode !== undefined ? Boolean(s.eco_mode) : true;
      state.ecoDelayS = Math.round(Number(s.eco_delay_s ?? 15));
      state.ecoFps = Number(s.eco_fps ?? 1.0);
      state.ecoMotionWake = s.eco_motion_wake !== undefined ? Boolean(s.eco_motion_wake) : true;
      state.ecoMotionFlash = s.eco_motion_flash !== undefined ? Boolean(s.eco_motion_flash) : true;
      state.ecoFlashDarkOnly = s.eco_flash_dark_only !== undefined ? Boolean(s.eco_flash_dark_only) : false;
      state.ecoMotionSensitivity = Math.round(Number(s.eco_motion_sensitivity ?? 50));
      state.ecoFlashDurationS = Math.round(Number(s.eco_flash_duration_s ?? 6));
      state.discordEnabled = s.discord_enabled !== undefined ? Boolean(s.discord_enabled) : true;
      state.discordCooldownS = Math.round(Number(s.discord_cooldown_s ?? 30));
      state.discordAbsenceS = Math.round(Number(s.discord_absence_reset_s ?? 15));
      state.discordIncludeScene = s.discord_include_scene !== undefined ? Boolean(s.discord_include_scene) : true;
      state.discordLogDogs = Boolean(s.discord_log_dogs);
      state.manualX = s.servo_x;
      state.manualY = s.servo_y;
      state.latchLaser = Boolean(s.manual_laser);
      state.latchLed = Boolean(s.manual_led);

      document.getElementById('detectPerson').checked = Boolean(s.detect_person);
      document.getElementById('detectDog').checked = Boolean(s.detect_dog);
      document.getElementById('detectInManual').checked = Boolean(s.detect_in_manual);
      document.getElementById('ecoMode').checked = state.ecoMode;
      document.getElementById('ecoDelay').value = state.ecoDelayS;
      document.getElementById('ecoFps').value = state.ecoFps;
      document.getElementById('ecoMotionWake').checked = state.ecoMotionWake;
      document.getElementById('ecoMotionFlash').checked = state.ecoMotionFlash;
      document.getElementById('ecoFlashDarkOnly').checked = state.ecoFlashDarkOnly;
      document.getElementById('ecoMotionSens').value = state.ecoMotionSensitivity;
      document.getElementById('ecoFlashDur').value = state.ecoFlashDurationS;
      document.getElementById('discordEnabled').checked = state.discordEnabled;
      document.getElementById('discordCooldown').value = state.discordCooldownS;
      document.getElementById('discordAbsence').value = state.discordAbsenceS;
      document.getElementById('discordIncludeScene').checked = state.discordIncludeScene;
      document.getElementById('discordLogDogs').checked = state.discordLogDogs;
      document.getElementById('autoFire').checked = Boolean(s.auto_fire);
      document.getElementById('ledMode').value = s.led_mode || 'on_lock';
      document.getElementById('kp').value = s.kp;
      document.getElementById('deadzone').value = s.deadzone_px;
      document.getElementById('conf').value = s.conf_threshold;
      document.getElementById('invX').checked = state.invertX;
      document.getElementById('invY').checked = state.invertY;
      const elCamBri = document.getElementById('camBri');
      const elCamAe = document.getElementById('camAe');
      const elCamGain = document.getElementById('camGain');
      const elCamNight = document.getElementById('camNight');
      if (elCamBri) elCamBri.value = state.camBri;
      if (elCamAe) elCamAe.value = state.camAe;
      if (elCamGain) elCamGain.value = state.camGain;
      if (elCamNight) elCamNight.checked = state.camNight;
      document.getElementById('mouseSens').value = state.mouseSensitivity;
      document.getElementById('joySens').value = state.joystickSpeedDegPerSec;
      document.getElementById('touchSens').value = state.touchAimSensitivity;

      const fireOut =
        state.autoFireLaser && state.autoFireLed ? 'both' : state.autoFireLed ? 'led' : 'laser';
      renderSegmented('#fireOutputSeg', 'val', fireOut);
      renderSegmented('#aimModeSeg', 'val', state.aimMode);
      renderSegmented('#gameAimModeSeg', 'val', state.aimMode);
      renderSegmented('#rotationSeg', 'rot', String(state.rotation));
      renderResolutionGrid(state.resolution);
      renderModeButtons();
      updateLabels();
      syncDrawerInputs();

      sf.syncFromServer({
        x: s.servo_x,
        y: s.servo_y,
        led: s.manual_led,
        laser: s.manual_laser,
      });

      state.initialized = true;
    } else if ((state.autoAim && s.auto_aim) || !recentlyControlled) {
      state.manualX = s.servo_x;
      state.manualY = s.servo_y;
      if (performance.now() - state.lastCamSentAt > 1200) {
        if (typeof s.resolution === 'number') {
          state.resolution = Math.max(0, Math.min(3, s.resolution));
          renderResolutionGrid(state.resolution);
        }
        let camLabelDirty = false;
        if (typeof s.cam_bri === 'number' && s.cam_bri !== state.camBri) {
          state.camBri = s.cam_bri;
          const el = document.getElementById('camBri');
          if (el) el.value = state.camBri;
          camLabelDirty = true;
        }
        if (typeof s.cam_ae === 'number' && s.cam_ae !== state.camAe) {
          state.camAe = s.cam_ae;
          const el = document.getElementById('camAe');
          if (el) el.value = state.camAe;
          camLabelDirty = true;
        }
        if (typeof s.cam_gain === 'number' && s.cam_gain !== state.camGain) {
          state.camGain = s.cam_gain;
          const el = document.getElementById('camGain');
          if (el) el.value = state.camGain;
          camLabelDirty = true;
        }
        if (s.cam_night !== undefined && Boolean(s.cam_night) !== state.camNight) {
          state.camNight = Boolean(s.cam_night);
          const el = document.getElementById('camNight');
          if (el) el.checked = state.camNight;
        }
        if (camLabelDirty) updateLabels();
      }
      if (!recentlyControlled && (performance.now() - state.lastConfigSentAt > 1200)) {
        if (state.autoAim !== Boolean(s.auto_aim)) {
          state.autoAim = Boolean(s.auto_aim);
          sf.sendAngles = !state.autoAim;
          renderModeButtons();
        }
        state.latchLaser = Boolean(s.manual_laser);
        state.latchLed = Boolean(s.manual_led);
        if (s.eco_mode !== undefined) {
          state.ecoMode = Boolean(s.eco_mode);
          const elEco = document.getElementById('ecoMode');
          if (elEco) elEco.checked = state.ecoMode;
          const gEco = document.getElementById('gameEcoMode');
          if (gEco) gEco.checked = state.ecoMode;
        }
        if (s.eco_motion_flash !== undefined) {
          state.ecoMotionFlash = Boolean(s.eco_motion_flash);
          const elMf = document.getElementById('ecoMotionFlash');
          if (elMf) elMf.checked = state.ecoMotionFlash;
          const gMf = document.getElementById('gameEcoMotionFlash');
          if (gMf) gMf.checked = state.ecoMotionFlash;
        }
        if (s.discord_enabled !== undefined) {
          state.discordEnabled = Boolean(s.discord_enabled);
          const elDc = document.getElementById('discordEnabled');
          if (elDc) elDc.checked = state.discordEnabled;
        }
      }
      sf.syncFromServer({
        x: s.servo_x,
        y: s.servo_y,
        led: s.manual_led,
        laser: s.manual_laser,
      });
    }

    if (state.autoAim && s.auto_aim) {
      document.getElementById('mAngles').textContent = `${s.servo_x}° / ${s.servo_y}°`;
    }

    renderManualVisuals(effectiveLaserForUi, effectiveLedForUi);
}

async function pollStatus() {
  if (document.hidden || statusPollState.inFlight) return;
  stopStatusPolling();
  statusPollState.inFlight = true;

  const controller = typeof AbortController === 'function' ? new AbortController() : null;
  const timeoutId = controller
    ? setTimeout(() => controller.abort(), STATUS_FETCH_TIMEOUT_MS)
    : null;

  try {
    const fetchOpts = { cache: 'no-store' };
    if (controller) fetchOpts.signal = controller.signal;
    const res = await fetch('/api/status', fetchOpts);
    if (!res.ok) {
      statusPollState.consecutiveErrors = Math.min(statusPollState.consecutiveErrors + 1, 5);
      return;
    }
    const s = await res.json();
    statusPollState.consecutiveErrors = 0;
    applyStatus(s);
  } catch (_err) {
    statusPollState.consecutiveErrors = Math.min(statusPollState.consecutiveErrors + 1, 5);
  } finally {
    if (timeoutId !== null) clearTimeout(timeoutId);
    statusPollState.inFlight = false;
    if (!document.hidden) {
      scheduleNextStatusPoll();
    }
  }
}

/* ================= MJPEG STREAM RECOVERY (MOBILE SLEEP/WAKE) ================= */
function setupStreamRecovery() {
  const img = document.getElementById('videoFeed');
  if (!img) return;
  let retryTimer = null;

  const refreshFeed = () => {
    if (retryTimer) clearTimeout(retryTimer);
    retryTimer = setTimeout(() => {
      img.src = `/video_feed?t=${Date.now()}`;
    }, 1200);
  };

  img.addEventListener('error', refreshFeed);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      releaseAllMomentaryInputs();
    } else if (!img.complete || img.naturalWidth === 0) {
      img.src = `/video_feed?t=${Date.now()}`;
    }
  });
}

/* ================= KEYBOARD SHORTCUTS ================= */
function onKeyDown(e) {
  if (e.repeat) return;
  const tag = (e.target && e.target.tagName) || '';
  if (!state.pointerLocked && (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA')) {
    return;
  }

  const k = e.key.toLowerCase();
  if (k === 'escape' && state.gameMode) {
    e.preventDefault();
    exitGameMode();
  } else if (k === 'm') {
    e.preventDefault();
    if (!state.pointerLocked) {
      enterPointerLock();
    }
  } else if (k === 'l') {
    e.preventDefault();
    toggleManualLaser();
  } else if (k === 'f') {
    e.preventDefault();
    toggleManualLed();
  } else if (k === 'c') {
    e.preventDefault();
    centerTurret();
  } else if (k === 'a' && !state.pointerLocked) {
    e.preventDefault();
    setOperatingMode(!state.autoAim);
  }
}

/* ================= UNLOAD FAIL-SAFE (LASER/FLASH OFF ON TAB CLOSE) ================= */
function sendUnloadFailsafe() {
  if (!getEffectiveManualLaser() && !getEffectiveManualLed()) {
    return;
  }
  releaseAllMomentaryInputs();
  state.latchLaser = false;
  state.latchLed = false;
  const payload = JSON.stringify({ action: 'failsafe' });
  if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
    try {
      const blob = new Blob([payload], { type: 'application/json' });
      if (navigator.sendBeacon('/api/action', blob)) {
        return;
      }
    } catch (_err) {}
  }
  try {
    fetch('/servo?laser=0&led=0', { method: 'GET', keepalive: true, cache: 'no-store' });
  } catch (_err) {}
}

/* ================= BOOTSTRAP ================= */
document.addEventListener('DOMContentLoaded', () => {
  initGimbalPad();
  bindJoystick('inlineJoystick', 'inlineJoystickKnob');
  bindJoystick('gameJoystick', 'gameJoystickKnob');
  initGameAimSurface();

  bindHoldActuatorButton('gameBtnFireLaser', 'laser');
  bindHoldActuatorButton('inlineBtnFireLaser', 'laser');
  bindHoldActuatorButton('btnPulseLaser', 'laser');
  bindHoldActuatorButton('btnPulseLed', 'led');

  setupStreamRecovery();

  const stage = document.getElementById('videoStage');
  if (stage) {
    stage.addEventListener('dblclick', (ev) => {
      if (state.gameMode || state.pointerLocked) return;
      if (ev.pointerType && ev.pointerType !== 'mouse') return;
      if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) return;
      enterPointerLock();
    });
    stage.addEventListener('contextmenu', (e) => {
      if (state.pointerLocked || state.gameMode) e.preventDefault();
    });
    stage.addEventListener('wheel', onPointerLockWheel, { passive: false });
  }

  const onFullscreenChange = () => {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    if (state.enteredNativeFs && !fsEl && state.gameMode) {
      exitGameMode({ skipExitFs: true });
    }
  };
  document.addEventListener('fullscreenchange', onFullscreenChange);
  document.addEventListener('webkitfullscreenchange', onFullscreenChange);

  document.addEventListener('pointerlockchange', onPointerLockChange);
  document.addEventListener('mousemove', onPointerLockMouseMove);
  document.addEventListener('mousedown', onPointerLockMouseDown);
  document.addEventListener('mouseup', onPointerLockMouseUp);
  document.addEventListener('keydown', onKeyDown);
  window.addEventListener('blur', releaseAllMomentaryInputs);
  window.addEventListener('pagehide', sendUnloadFailsafe);
  window.addEventListener('beforeunload', sendUnloadFailsafe);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      stopStatusPolling();
    } else {
      statusPollState.consecutiveErrors = 0;
      pollStatus();
    }
  });

  updateLabels();
  renderManualVisuals();
  if (!document.hidden) {
    pollStatus();
  }
});
