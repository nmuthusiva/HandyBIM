import * as THREE from 'three';
import * as OBC from '@thatopen/components';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { AudioManager } from './audioManager.js';
import { HandControl, GestureState } from './handControl';

// Must match the installed web-ifc version (node_modules/web-ifc/package.json) —
// see the comment in _setupIfc() for why this can't be auto-detected.
const WEB_IFC_VERSION = '0.0.77';

// Control-value -> scene-transform sensitivities. HandControl emits normalized,
// application-agnostic deltas; these tune how "big" they feel in this viewer.
const ROTATE_SENSITIVITY = 6; // radians per unit of normalized dx/dy
const PAN_SENSITIVITY = 1; // world-pixels per unit of normalized dx/dy * viewport size
const SECTION_SENSITIVITY = 2; // world-pixels per unit of normalized delta * viewport height
const ZOOM_MIN_FACTOR = 0.2;
const ZOOM_MAX_FACTOR = 5;

const SECTION_ORIENTATIONS = [
  { label: 'Horizontal', normal: new THREE.Vector3(0, 1, 0) },
  { label: 'Vertical X', normal: new THREE.Vector3(1, 0, 0) },
  { label: 'Vertical Z', normal: new THREE.Vector3(0, 0, 1) },
];

const GESTURE_STATE_COLORS = {
  [GestureState.IDLE]: '#888888',
  [GestureState.HAND_VISIBLE]: '#aaaaaa',
  [GestureState.POINTING]: '#00BFFF',
  [GestureState.PINCHING]: '#00FFFF',
  [GestureState.ROTATING]: '#00FFFF',
  [GestureState.PANNING]: '#FF00FF',
  [GestureState.TWO_HAND_ZOOM]: '#FFFF00',
  [GestureState.SECTION_ACTIVE]: '#FFA500',
};

const GESTURE_STATE_LABELS = {
  [GestureState.IDLE]: 'No hand',
  [GestureState.HAND_VISIBLE]: 'Hand ready',
  [GestureState.POINTING]: 'Pointing',
  [GestureState.PINCHING]: 'Pinch held',
  [GestureState.ROTATING]: 'Rotating',
  [GestureState.PANNING]: 'Panning',
  [GestureState.TWO_HAND_ZOOM]: 'Two-hand zoom',
  [GestureState.SECTION_ACTIVE]: 'Section cut',
  [GestureState.PAUSE_TOGGLE]: 'Pausing gestures',
  [GestureState.RESET_VIEW]: 'Resetting view',
};

const SEGMENTER_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite';
const SEGMENTER_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm';
const BACKGROUND_BLUR_PX = 24;

export class Game {
  constructor(renderDiv) {
    this.renderDiv = renderDiv;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.videoElement = null;
    this.backgroundVisible = true;
    this.backgroundBlurred = false;
    this.backgroundCanvas = null;
    this.backgroundSourceCanvas = null;
    this.backgroundMaskCanvas = null;
    this.backgroundSegmenter = null;
    this.backgroundSegmenterLoading = false;
    this.lastSegmentationTime = -1;

    this.gameState = 'loading'; // loading, tracking, error
    this.clock = new THREE.Clock();
    this.audioManager = new AudioManager();

    // IFC loading (That Open Components / Fragments)
    this.obcComponents = null;
    this.fragments = null;
    this.ifcLoader = null;
    this.currentModel = null; // THREE.Group wrapping the current fragments model, gesture target
    this.currentFragmentsModel = null; // FRAGS.FragmentsModel, for disposal
    this.currentFragmentsModelId = null;
    this.baseModelScale = 1;

    this.modelMinZ = -200;
    this.modelMaxZ = 50;
    this.modelDefaultZ = -100;

    // Hand control (standalone gesture engine, see src/handControl)
    this.handControl = null;

    // Section / clipping plane
    this.sectionPlane = new THREE.Plane(SECTION_ORIENTATIONS[0].normal.clone(), 0);
    this.sectionOrientationIndex = 0;
    this.sectionOffset = 0;
    this.sectionEnabled = false;
    this.sectionPlaneVisual = null;

    // Element selection & properties inspector
    this.selectedLocalId = null;
    this.selectionBoxHelper = null;
    this.propertiesPanel = null;
    this.cursorPos = { x: 0.5, y: 0.5 };

    // Dwell (one-finger hold to select) settings
    this.dwellAnchorPos = null;
    this.dwellStartTime = 0;
    this.dwellTriggered = false;
    this.dwellDuration = 800; // 800ms hold to select (faster response)
    this.dwellTolerance = 32; // pixels tolerance for natural hand tremor
    this.dwellCircle = null;
    this.cursorRingCircumference = 2 * Math.PI * 18; // ~113.1

    // Panel finger move & scroll settings
    this.panelAttachedToFinger = false;
    this.isPointerOverPanel = false;
    this.isPointerOverPanelHeader = false;
    this.isPinchingPanelHeader = false;
    this.scrollHintEl = null;

    this._eventsBound = false;
    this._animating = false;

    this._init().catch((error) => {
      console.error('Initialization failed:', error);
      const detail = error?.message || String(error);
      this._showError(`Initialization failed: ${detail}`, true);
    });
  }

  async _init() {
    this._setupDOM();
    this._setupThree();

    // 1. Get webcam running first so the user gets immediate video feedback
    try {
      await this._setupWebcam();
    } catch (error) {
      console.error('Camera access failed:', error);
      let message = `Camera error: ${error.message || error}`;
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        message = 'Camera access was blocked or denied. Please click the camera icon in your browser address bar to allow camera access.';
      } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        message = 'No webcam was detected. Please connect a camera and try again.';
      } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        message = 'Webcam is in use by another application (e.g., Teams/Zoom) or blocked by Windows Camera Privacy settings.';
      }
      this._showError(message, true, true);
      return;
    }

    // 2. Setup IFC engine
    try {
      await this._setupIfc();
    } catch (ifcErr) {
      console.error('IFC setup failed:', ifcErr);
      this._showError(`IFC engine initialization failed: ${ifcErr.message || ifcErr}`, true);
      return;
    }

    // 3. Background segmenter (resilient fallback)
    try {
      await this._setupBackgroundSegmenter();
    } catch (bgErr) {
      console.warn('Background segmenter setup error (non-fatal):', bgErr);
    }

    // 4. Hand tracking (resilient fallback)
    try {
      await this._setupHandControl();
    } catch (handErr) {
      console.warn('Hand tracking setup error (falling back to keyboard controls):', handErr);
      if (this.gestureBadge) {
        this.gestureBadge.innerText = 'Gestures offline';
        this.gestureBadge.style.backgroundColor = '#ff6b6b';
      }
    }

    this.audioManager.resumeContext();
    this.clock.start();
    if (!this._eventsBound) {
      window.addEventListener('resize', this._onResize.bind(this));
      window.addEventListener('keydown', this._onKeyDown.bind(this));
      this._eventsBound = true;
    }

    this.gameState = 'tracking';
    this._updateStatusOverlayVisibility();
    if (!this._animating) {
      this._animating = true;
      this._animate();
    }
  }

  // ---------------------------------------------------------------------
  // DOM / UI setup
  // ---------------------------------------------------------------------

  _setupDOM() {
    if (this.videoElement) return;
    this.videoElement = document.createElement('video');
    Object.assign(this.videoElement.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: '100%',
      height: '100%',
      objectFit: 'cover',
      transform: 'scaleX(-1)', // mirror for intuitive control
      zIndex: '0',
    });
    this.videoElement.autoplay = true;
    this.videoElement.muted = true;
    this.videoElement.playsInline = true;
    this.renderDiv.appendChild(this.videoElement);

    this.backgroundCanvas = document.createElement('canvas');
    this.backgroundCanvas.className = 'background-canvas';
    this.backgroundCanvas.style.display = 'none';
    this.backgroundSourceCanvas = document.createElement('canvas');
    this.backgroundMaskCanvas = document.createElement('canvas');
    this.renderDiv.appendChild(this.backgroundCanvas);

    this._setupStatusOverlay();
    this._setupGestureBadge();
    this._setupGestureControls();
    this._setupLandmarkOverlay();
    this._setupCursorDot();
    this._setupPropertiesPanel();
    this._setupLegend();
    this._setupDragAndDrop();
    this._setupMouseControls();
  }

  _setupStatusOverlay() {
    this.statusContainer = document.createElement('div');
    Object.assign(this.statusContainer.style, {
      position: 'absolute',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: '10',
      pointerEvents: 'none',
      textAlign: 'center',
      color: 'white',
      fontFamily: '"Arial", "Helvetica Neue", Helvetica, sans-serif',
      maxWidth: '80vw',
    });

    this.statusText = document.createElement('div');
    this.statusText.style.fontSize = 'clamp(20px, 4vw, 36px)';
    this.statusText.style.fontWeight = 'bold';
    this.statusText.style.textShadow = '2px 2px 4px black';
    this.statusText.innerText = 'Drag & drop an .ifc file to load a model';
    this.statusContainer.appendChild(this.statusText);

    this.renderDiv.appendChild(this.statusContainer);
  }

  _setupGestureBadge() {
    this.gestureBadge = document.createElement('div');
    Object.assign(this.gestureBadge.style, {
      position: 'absolute',
      top: '10px',
      right: '10px',
      zIndex: '30',
      padding: '10px 18px',
      fontFamily: '"Arial", "Helvetica Neue", Helvetica, sans-serif',
      fontWeight: 'bold',
      fontSize: '16px',
      border: '2px solid black',
      borderRadius: '4px',
      boxShadow: '2px 2px 0px black',
      backgroundColor: '#888888',
      color: '#000000',
      textAlign: 'center',
    });
    this.gestureBadge.innerText = GestureState.IDLE;
    this.renderDiv.appendChild(this.gestureBadge);

    this.sectionBadge = document.createElement('div');
    Object.assign(this.sectionBadge.style, {
      position: 'absolute',
      top: '54px',
      right: '10px',
      zIndex: '30',
      padding: '8px 14px',
      fontFamily: '"Arial", "Helvetica Neue", Helvetica, sans-serif',
      fontWeight: 'bold',
      fontSize: '14px',
      border: '2px solid black',
      borderRadius: '4px',
      boxShadow: '2px 2px 0px black',
      backgroundColor: 'white',
      color: '#000000',
      textAlign: 'center',
    });
    this._updateSectionBadgeText();
    this.renderDiv.appendChild(this.sectionBadge);
  }

  _setupGestureControls() {
    this.gestureControls = document.createElement('div');
    this.gestureControls.className = 'gesture-controls';

    this.pauseButton = document.createElement('button');
    this.pauseButton.type = 'button';
    this.pauseButton.innerText = 'Pause gestures';
    this.pauseButton.title = 'Pause or resume hand tracking (Space)';
    this.pauseButton.addEventListener('click', () => this._toggleGesturePause());

    this.resetButton = document.createElement('button');
    this.resetButton.type = 'button';
    this.resetButton.innerText = 'Reset view';
    this.resetButton.title = 'Reset model position, rotation, zoom, and section (R)';
    this.resetButton.addEventListener('click', () => this._resetView());

    this.backgroundButton = document.createElement('button');
    this.backgroundButton.type = 'button';
    this.backgroundButton.title = 'Cycle camera background: clear, blurred, or hidden (B)';
    this.backgroundButton.addEventListener('click', () => this._toggleBackground());

    this.landmarkButton = document.createElement('button');
    this.landmarkButton.type = 'button';
    this.landmarkButton.title = 'Show or hide finger movement lines (L)';
    this.landmarkButton.addEventListener('click', () => this._toggleLandmarkOverlay());

    this.infoButton = document.createElement('button');
    this.infoButton.type = 'button';
    this.infoButton.innerText = 'ⓘ Info';
    this.infoButton.title = 'Show or hide the gesture legend (I)';
    this.infoButton.setAttribute('aria-pressed', 'false');
    this.infoButton.addEventListener('click', () => this._toggleInfoPanel());

    this.gestureControls.append(
      this.pauseButton,
      this.resetButton,
      this.backgroundButton,
      this.landmarkButton,
      this.infoButton
    );
    this._updateBackgroundButton();
    this._updateLandmarkButton();
    this.renderDiv.appendChild(this.gestureControls);

    document.addEventListener('click', (event) => {
      if (!this.infoPanelOpen) return;
      if (event.target === this.infoButton || this.legendText?.contains(event.target)) return;
      this._closeInfoPanel();
    });
  }

  _setupCursorDot() {
    this.cursorDot = document.createElement('div');
    this.cursorDot.className = 'dwell-cursor-container';
    Object.assign(this.cursorDot.style, {
      position: 'absolute',
      top: '0',
      left: '0',
      width: '44px',
      height: '44px',
      marginLeft: '-22px',
      marginTop: '-22px',
      zIndex: '35',
      pointerEvents: 'none',
      display: 'none',
      transition: 'transform 0.12s ease-out, filter 0.15s ease-out',
    });

    const circumference = this.cursorRingCircumference;
    this.cursorDot.innerHTML = `
      <svg width="44" height="44" viewBox="0 0 44 44" style="transform: rotate(-90deg); position: absolute; top: 0; left: 0;">
        <circle cx="22" cy="22" r="18" fill="rgba(10, 20, 35, 0.45)" stroke="rgba(255, 255, 255, 0.35)" stroke-width="2.5" />
        <circle class="dwell-progress-ring" cx="22" cy="22" r="18" fill="none" stroke="#00ffff" stroke-width="3.5"
                stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}" stroke-linecap="round"
                style="transition: stroke-dashoffset 0.04s linear;" />
      </svg>
      <div class="dwell-center-dot" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 10px; height: 10px; border-radius: 50%; background: #ffffff; box-shadow: 0 0 8px #00ffff, 0 0 14px rgba(0, 255, 255, 0.8);"></div>
    `;

    this.renderDiv.appendChild(this.cursorDot);
    this.dwellCircle = this.cursorDot.querySelector('.dwell-progress-ring');
  }

  _setupPropertiesPanel() {
    if (this.propertiesPanel) return;
    this.propertiesPanel = document.createElement('div');
    this.propertiesPanel.className = 'properties-panel hidden';
    this.propertiesPanel.id = 'properties-panel';
    this.renderDiv.appendChild(this.propertiesPanel);
    this._setupPanelDragging();
  }

  _setupPanelDragging() {
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;
    let activeHeader = null;

    const onPointerDown = (e) => {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      const header = e.target.closest('.properties-header');
      if (!header || e.target.closest('button, a')) return;

      if (this.panelAttachedToFinger) {
        this._togglePanelFingerMove();
        return;
      }

      isDragging = true;
      activeHeader = header;
      startX = e.clientX;
      startY = e.clientY;

      const rect = this.propertiesPanel.getBoundingClientRect();
      startLeft = rect.left;
      startTop = rect.top;

      this.propertiesPanel.style.left = `${startLeft}px`;
      this.propertiesPanel.style.top = `${startTop}px`;
      this.propertiesPanel.style.right = 'auto';
      this.propertiesPanel.style.bottom = 'auto';

      header.style.cursor = 'grabbing';
      document.body.style.userSelect = 'none';

      try {
        header.setPointerCapture(e.pointerId);
      } catch (_) {}
    };

    const onPointerMove = (e) => {
      if (!isDragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;

      const panelWidth = this.propertiesPanel.offsetWidth;
      const panelHeight = this.propertiesPanel.offsetHeight;
      const maxLeft = window.innerWidth - panelWidth - 8;
      const maxTop = window.innerHeight - panelHeight - 8;

      const nextLeft = Math.max(8, Math.min(maxLeft, startLeft + dx));
      const nextTop = Math.max(8, Math.min(maxTop, startTop + dy));

      this.propertiesPanel.style.left = `${nextLeft}px`;
      this.propertiesPanel.style.top = `${nextTop}px`;
    };

    const onPointerUp = (e) => {
      if (!isDragging) return;
      isDragging = false;
      if (activeHeader) {
        activeHeader.style.cursor = 'grab';
        try {
          activeHeader.releasePointerCapture(e.pointerId);
        } catch (_) {}
        activeHeader = null;
      }
      document.body.style.userSelect = '';
    };

    this.propertiesPanel.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
  }

  _setupLegend() {
    this.legendText = document.querySelector('#instruction-text');
    if (!this.legendText) return;
    this.legendText.innerHTML =
      'Point & hold (0.8s): select element or button &nbsp;·&nbsp; Hold title bar: move / drop panel &nbsp;·&nbsp; ' +
      'Touch ⊞ Expand: open/close all subpanels &nbsp;·&nbsp; Touch ▲/▼: scroll properties &nbsp;·&nbsp; ' +
      'Pinch+drag: rotate &nbsp;·&nbsp; Open palm: pan &nbsp;·&nbsp; Two hands: zoom &nbsp;·&nbsp; ✌ two fingers: section cut &nbsp;·&nbsp; Fist: reset &nbsp;·&nbsp; ' +
      'Mouse: drag header / wheel to scroll';
    this.infoPanelOpen = false;
    this.legendText.style.display = 'none';
  }

  _toggleInfoPanel() {
    if (this.infoPanelOpen) this._closeInfoPanel();
    else this._openInfoPanel();
  }

  _openInfoPanel() {
    if (!this.legendText) return;
    this.infoPanelOpen = true;
    this.legendText.style.display = 'block';
    this.infoButton?.setAttribute('aria-pressed', 'true');
  }

  _closeInfoPanel() {
    if (!this.legendText) return;
    this.infoPanelOpen = false;
    this.legendText.style.display = 'none';
    this.infoButton?.setAttribute('aria-pressed', 'false');
  }

  _setupDragAndDrop() {
    this.renderDiv.addEventListener('dragover', (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
      this.renderDiv.style.border = '2px dashed #007bff';
    });
    this.renderDiv.addEventListener('dragleave', () => {
      this.renderDiv.style.border = 'none';
    });
    this.renderDiv.addEventListener('drop', (event) => {
      event.preventDefault();
      this.renderDiv.style.border = 'none';
      if (!event.dataTransfer.files || event.dataTransfer.files.length === 0) return;

      const file = event.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith('.ifc')) {
        this._loadIfcFile(file);
      } else {
        this._showStatusScreen(`"${file.name}" is not an IFC model.`, 'orange');
        setTimeout(() => this._updateStatusOverlayVisibility(), 3000);
      }
      event.dataTransfer.clearData();
    });
  }

  _setupMouseControls() {
    let isDragging = false;
    let dragButton = 0;
    let startX = 0;
    let startY = 0;
    let prevX = 0;
    let prevY = 0;

    this.renderDiv.addEventListener('mousedown', (e) => {
      if (e.target !== this.renderDiv && e.target !== this.renderer?.domElement && e.target !== this.videoElement) return;
      isDragging = true;
      dragButton = e.button;
      startX = e.clientX;
      startY = e.clientY;
      prevX = e.clientX;
      prevY = e.clientY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!isDragging || !this.currentModel) return;
      const dx = (e.clientX - prevX) / this.renderDiv.clientWidth;
      const dy = (e.clientY - prevY) / this.renderDiv.clientHeight;
      prevX = e.clientX;
      prevY = e.clientY;

      if (dragButton === 0) {
        this._onRotate(dx, dy);
      } else if (dragButton === 2) {
        this._onPan(dx, dy);
      }
    });

    window.addEventListener('mouseup', (e) => {
      if (!isDragging) return;
      isDragging = false;
      const dist = Math.hypot(e.clientX - startX, e.clientY - startY);
      if (dragButton === 0 && dist < 6) {
        this._onSelect(e.clientX, e.clientY, false);
      }
    });

    this.renderDiv.addEventListener(
      'wheel',
      (e) => {
        if (e.target.closest('.properties-panel')) {
          // Allow natural mouse-wheel scrolling inside the properties panel
          return;
        }
        if (!this.currentModel) return;
        e.preventDefault();
        const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
        this._onZoom(zoomFactor);
      },
      { passive: false }
    );

    this.renderDiv.addEventListener('contextmenu', (e) => {
      if (this.currentModel) e.preventDefault();
    });
  }

  _setupThree() {
    if (this.renderer) return;
    const width = this.renderDiv.clientWidth;
    const height = this.renderDiv.clientHeight;

    this.scene = new THREE.Scene();

    // Orthographic camera with 1 world-unit == 1 screen pixel, matching the
    // normalized-to-pixel control math applied to gesture deltas.
    this.camera = new THREE.OrthographicCamera(width / -2, width / 2, height / 2, height / -2, 1, 2000);
    this.camera.position.z = 100;

    this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.localClippingEnabled = true;
    Object.assign(this.renderer.domElement.style, { position: 'absolute', top: '0', left: '0', zIndex: '1' });
    this.renderDiv.appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.8);
    directionalLight.position.set(0, 0, 100);
    this.scene.add(directionalLight);

    const visualGeometry = new THREE.PlaneGeometry(4000, 4000);
    const visualMaterial = new THREE.MeshBasicMaterial({
      color: 0xffa500,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.sectionPlaneVisual = new THREE.Mesh(visualGeometry, visualMaterial);
    this.sectionPlaneVisual.visible = false;
    this.scene.add(this.sectionPlaneVisual);
    this._rebuildSectionPlane();
  }

  // ---------------------------------------------------------------------
  // IFC loading (That Open Components)
  // ---------------------------------------------------------------------

  async _setupIfc() {
    this.obcComponents = new OBC.Components();

    this.fragments = this.obcComponents.get(OBC.FragmentsManager);
    const workerUrl = await OBC.FragmentsManager.getWorker();
    this.fragments.init(workerUrl);

    this.ifcLoader = this.obcComponents.get(OBC.IfcLoader);
    await this.ifcLoader.setup({ autoSetWasm: false });
    this.ifcLoader.settings.wasm.path = `https://unpkg.com/web-ifc@${WEB_IFC_VERSION}/`;
    this.ifcLoader.settings.wasm.absolute = true;

    this.obcComponents.init();
  }

  async _loadIfcFile(file) {
    this._showStatusScreen(`Loading "${file.name}"...`, 'white');
    try {
      const buffer = await file.arrayBuffer();
      await this._parseAndLoadIfc(buffer, file.name);
      this._updateStatusOverlayVisibility();
    } catch (error) {
      console.error(`Error loading IFC model ${file.name}:`, error);
      this._showStatusScreen(`Failed to load "${file.name}". Check console for details.`, 'orange');
    }
  }

  async _parseAndLoadIfc(arrayBuffer, fileName) {
    this._disposeCurrentModel();

    const modelId = `${fileName}-${Date.now()}`;
    const data = new Uint8Array(arrayBuffer);
    const fragmentsModel = await this.ifcLoader.load(data, true, modelId);

    // Fragments streams tile geometry only for what's inside the registered
    // camera's frustum. Real-world IFC files often place geometry far from
    // (0,0,0) (survey/site coordinates), which our small display frustum
    // wouldn't see — so discover the true bounding box with an effectively
    // unbounded camera first, before anything has been culled out.
    const discoveryCamera = new THREE.OrthographicCamera(-1e8, 1e8, 1e8, -1e8, -1e8, 1e8);
    fragmentsModel.useCamera(discoveryCamera);
    await this.fragments.core.update(true);

    this.currentFragmentsModel = fragmentsModel;
    this.currentFragmentsModelId = modelId;
    this.currentModel = this._fitModelToView(fragmentsModel);
    this._applySectionClipping(fragmentsModel.object);
    this.scene.add(this.currentModel);

    // Now that the model's world position sits inside the real display
    // camera's frustum, re-stream tiles against that camera for rendering.
    fragmentsModel.useCamera(this.camera);
    await this.fragments.core.update(true);

    // Expand fragmentsModel.box so that raycast's frustum.intersectsBox check
    // is never culled out due to raw vs display coordinate differences.
    if (fragmentsModel.box) {
      fragmentsModel.box.min.set(-1e9, -1e9, -1e9);
      fragmentsModel.box.max.set(1e9, 1e9, 1e9);
    }
  }

  // Centers and uniformly scales the loaded model so it fits comfortably in
  // the ortho camera's screen-pixel frustum, regardless of the model's
  // original real-world units.
  _fitModelToView(fragmentsModel) {
    const object = fragmentsModel.object;
    const box = fragmentsModel.box?.isEmpty?.() === false
      ? fragmentsModel.box
      : new THREE.Box3().setFromObject(object);

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    object.position.sub(center); // re-center object's own content at local origin

    const maxDimension = Math.max(size.x, size.y, size.z, 0.0001);
    const targetPixelSize = Math.min(this.renderDiv.clientWidth, this.renderDiv.clientHeight) * 0.6;
    const scale = targetPixelSize / maxDimension;
    this.baseModelScale = scale;

    const wrapper = new THREE.Group();
    wrapper.add(object);
    wrapper.scale.set(scale, scale, scale);
    wrapper.position.set(0, 0, this.modelDefaultZ);
    return wrapper;
  }

  _applySectionClipping(object) {
    object.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        material.clippingPlanes = this.sectionEnabled ? [this.sectionPlane] : [];
        material.clipShadows = true;
        material.needsUpdate = true;
      });
    });
  }

  _disposeCurrentModel() {
    this._deselectElement();
    if (this.currentModel) {
      this.scene.remove(this.currentModel);
      this.currentModel = null;
    }
    if (this.currentFragmentsModelId) {
      this.fragments.core.disposeModel(this.currentFragmentsModelId).catch((e) => {
        console.warn('Error disposing previous IFC model:', e);
      });
      this.currentFragmentsModelId = null;
    }
    this.currentFragmentsModel = null;
  }

  _updateStatusOverlayVisibility() {
    const actionsWrapper = document.getElementById('init-actions-wrapper');
    if (actionsWrapper) actionsWrapper.remove();
    const retryBtn = document.getElementById('retry-init-btn');
    if (retryBtn) retryBtn.remove();

    if (this.currentModel) {
      this.statusContainer.style.display = 'none';
    } else {
      this.statusContainer.style.display = 'block';
      this.statusText.innerText = 'Drag & drop an .ifc file to load a model';
      this.statusText.style.color = 'white';
    }
  }

  _showStatusScreen(message, color = 'white') {
    this.statusContainer.style.display = 'block';
    this.statusText.innerText = message;
    this.statusText.style.color = color;
  }

  // ---------------------------------------------------------------------
  // Webcam + hand control
  // ---------------------------------------------------------------------

  async _setupWebcam() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error(
        'Camera API not supported or inaccessible. Ensure you are visiting via http://localhost or HTTPS.'
      );
    }

    let stream = null;
    try {
      // First try user-facing HD stream
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
    } catch (constraintErr) {
      console.warn('Ideal camera constraints failed, attempting fallback to basic video:', constraintErr);
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    this.videoElement.srcObject = stream;

    // Immediately play the video so feed is live
    try {
      await this.videoElement.play();
    } catch (playErr) {
      console.warn('Autoplay may have been deferred by browser:', playErr);
    }

    // Wait for video metadata to have valid dimensions, with a safe 3s fallback
    if (this.videoElement.readyState < 1) {
      await new Promise((resolve) => {
        const onMetadata = () => {
          this.videoElement.removeEventListener('loadedmetadata', onMetadata);
          resolve();
        };
        this.videoElement.addEventListener('loadedmetadata', onMetadata);
        setTimeout(resolve, 3000);
      });
    }

    this.videoElement.style.width = `${this.renderDiv.clientWidth}px`;
    this.videoElement.style.height = `${this.renderDiv.clientHeight}px`;
  }

  async _setupBackgroundSegmenter() {
    if (this.backgroundSegmenterLoading || this.backgroundSegmenter) return;
    this.backgroundSegmenterLoading = true;
    this._updateBackgroundButton();
    try {
      const vision = await FilesetResolver.forVisionTasks(SEGMENTER_WASM_PATH);
      try {
        this.backgroundSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: SEGMENTER_MODEL_PATH, delegate: 'GPU' },
          runningMode: 'VIDEO',
          outputCategoryMask: true,
          outputConfidenceMasks: false,
        });
      } catch (gpuError) {
        console.warn('GPU delegate failed for ImageSegmenter, falling back to CPU:', gpuError);
        this.backgroundSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: SEGMENTER_MODEL_PATH, delegate: 'CPU' },
          runningMode: 'VIDEO',
          outputCategoryMask: true,
          outputConfidenceMasks: false,
        });
      }
    } catch (error) {
      console.warn('Person-aware blur unavailable; using full-camera blur fallback.', error);
      this.backgroundSegmenter = null;
      this.backgroundCanvas.style.display = 'none';
      if (this.backgroundBlurred) this.videoElement.style.filter = `blur(${BACKGROUND_BLUR_PX}px)`;
    } finally {
      this.backgroundSegmenterLoading = false;
      this._updateBackgroundButton();
    }
  }

  async _setupHandControl() {
    this.handControl = new HandControl({
      videoElement: this.videoElement,
      maxHands: 2,
      smoothing: 0.2,
      mirrorX: true,
    });

    this.handControl.on('rotate', ({ dx, dy, x, y }) => this._onRotate(dx, dy, x, y));
    this.handControl.on('zoom', ({ scale }) => this._onZoom(scale));
    this.handControl.on('pan', ({ dx, dy, x, y }) => this._onPan(dx, dy, x, y));
    this.handControl.on('pointer', ({ x, y }) => this._onPointer(x, y));
    this.handControl.on('select', ({ x, y }) => this._onSelect(x, y));
    this.handControl.on('sectionStart', () => this._onSectionStart());
    this.handControl.on('sectionMove', ({ delta }) => this._onSectionMove(delta));
    this.handControl.on('sectionEnd', () => this._onSectionEnd());
    this.handControl.on('pauseToggle', () => this._toggleGesturePause());
    this.handControl.on('resetView', () => this._resetView());
    this.handControl.on('landmarks', (hands) => this._drawLandmarks(hands));

    await this.handControl.start();
  }

  // ---------------------------------------------------------------------
  // Gesture -> scene wiring
  // ---------------------------------------------------------------------

  _onRotate(dx, dy, x, y) {
    // If pinch is over the panel header, drag the panel with your hand!
    if (this.propertiesPanel && !this.propertiesPanel.classList.contains('hidden')) {
      let isOverHeader = this.isPointerOverPanelHeader;
      if (x !== undefined && y !== undefined) {
        const screenPoint = this._landmarkToOverlay({ x, y });
        const header = this.propertiesPanel.querySelector('.properties-header');
        if (header) {
          const headerRect = header.getBoundingClientRect();
          isOverHeader = (
            screenPoint.x >= headerRect.left && screenPoint.x <= headerRect.right &&
            screenPoint.y >= headerRect.top && screenPoint.y <= headerRect.bottom
          );
        }
      }

      if (isOverHeader) {
        const panelWidth = this.propertiesPanel.offsetWidth;
        const panelHeight = this.propertiesPanel.offsetHeight;
        const maxLeft = window.innerWidth - panelWidth - 10;
        const maxTop = window.innerHeight - panelHeight - 10;

        const currentLeft = this.propertiesPanel.offsetLeft;
        const currentTop = this.propertiesPanel.offsetTop;

        const newLeft = Math.max(10, Math.min(maxLeft, currentLeft + dx * this.renderDiv.clientWidth * 1.5));
        const newTop = Math.max(10, Math.min(maxTop, currentTop + dy * this.renderDiv.clientHeight * 1.5));

        this.propertiesPanel.style.left = `${newLeft}px`;
        this.propertiesPanel.style.top = `${newTop}px`;
        this.propertiesPanel.style.right = 'auto';
        this.propertiesPanel.style.bottom = 'auto';
        return;
      }
    }

    if (!this.currentModel) return;
    this.currentModel.rotation.y -= dx * ROTATE_SENSITIVITY;
    this.currentModel.rotation.x -= dy * ROTATE_SENSITIVITY;
    this._rebuildSectionPlane();
    if (this.selectedLocalId !== null) {
      this._updateSelectionHighlightBox(this.selectedLocalId);
    }
  }

  _onZoom(scale) {
    if (!this.currentModel) return;
    const current = this.currentModel.scale.x;
    const next = Math.max(
      this.baseModelScale * ZOOM_MIN_FACTOR,
      Math.min(this.baseModelScale * ZOOM_MAX_FACTOR, current * scale)
    );
    this.currentModel.scale.set(next, next, next);
    if (this.selectedLocalId !== null) {
      this._updateSelectionHighlightBox(this.selectedLocalId);
    }
  }

  _onPan(dx, dy, x, y) {
    // If open palm is over properties panel, air-scroll the properties content!
    if (this.propertiesPanel && !this.propertiesPanel.classList.contains('hidden')) {
      let isOverPanel = this.isPointerOverPanel;
      if (x !== undefined && y !== undefined) {
        const screenPoint = this._landmarkToOverlay({ x, y });
        const panelRect = this.propertiesPanel.getBoundingClientRect();
        isOverPanel = (
          screenPoint.x >= panelRect.left && screenPoint.x <= panelRect.right &&
          screenPoint.y >= panelRect.top && screenPoint.y <= panelRect.bottom
        );
      }

      if (isOverPanel) {
        const content = this.propertiesPanel.querySelector('.properties-content');
        if (content) {
          // Touch/air swipe: move hand up to scroll down, move hand down to scroll up
          content.scrollTop -= dy * this.renderDiv.clientHeight * 1.5;
          return;
        }
      }
    }

    if (!this.currentModel) return;
    this.currentModel.position.x += dx * this.renderDiv.clientWidth * PAN_SENSITIVITY;
    this.currentModel.position.y -= dy * this.renderDiv.clientHeight * PAN_SENSITIVITY;
    this._rebuildSectionPlane();
    if (this.selectedLocalId !== null) {
      this._updateSelectionHighlightBox(this.selectedLocalId);
    }
  }

  _onPointer(x, y) {
    this.cursorPos = { x, y };
    const screenPoint = this._landmarkToOverlay({ x, y });
    const screenX = screenPoint.x;
    const screenY = screenPoint.y;

    this.cursorDot.style.display = 'block';
    this.cursorDot.style.left = `${screenX}px`;
    this.cursorDot.style.top = `${screenY}px`;

    // 1. Move panel attached to finger
    if (this.panelAttachedToFinger && this.propertiesPanel && !this.propertiesPanel.classList.contains('hidden')) {
      const panelWidth = this.propertiesPanel.offsetWidth;
      const panelHeight = this.propertiesPanel.offsetHeight;
      const maxLeft = window.innerWidth - panelWidth - 10;
      const maxTop = window.innerHeight - panelHeight - 10;

      const newLeft = Math.max(10, Math.min(maxLeft, screenX - 60));
      const newTop = Math.max(10, Math.min(maxTop, screenY - 20));

      this.propertiesPanel.style.left = `${newLeft}px`;
      this.propertiesPanel.style.top = `${newTop}px`;
      this.propertiesPanel.style.right = 'auto';
    }

    // 2. Track pointer relative to panel and auto-scroll
    const hoveredEl = document.elementFromPoint(screenX, screenY);
    if (this.propertiesPanel && !this.propertiesPanel.classList.contains('hidden')) {
      const panelRect = this.propertiesPanel.getBoundingClientRect();
      this.isPointerOverPanel = (
        screenX >= panelRect.left && screenX <= panelRect.right &&
        screenY >= panelRect.top && screenY <= panelRect.bottom
      );

      const header = this.propertiesPanel.querySelector('.properties-header');
      if (header) {
        const headerRect = header.getBoundingClientRect();
        this.isPointerOverPanelHeader = (
          screenX >= headerRect.left && screenX <= headerRect.right &&
          screenY >= headerRect.top && screenY <= headerRect.bottom
        );
      } else {
        this.isPointerOverPanelHeader = false;
      }

      // Continuous finger edge auto-scrolling
      const content = this.propertiesPanel.querySelector('.properties-content');
      if (this.isPointerOverPanel && content && !this.panelAttachedToFinger) {
        const contentRect = content.getBoundingClientRect();
        const isInContent = (
          screenX >= contentRect.left && screenX <= contentRect.right &&
          screenY >= contentRect.top && screenY <= contentRect.bottom
        );

        if (isInContent) {
          const distFromBottom = contentRect.bottom - screenY;
          const distFromTop = screenY - contentRect.top;

          // Don't auto-scroll if hovering over an interactive item (button or pset header)
          const isOverContentItem = hoveredEl && (hoveredEl.closest('button, a, .pset-header, .top-scroll-btn, .top-nav-btn') !== null);

          if (!isOverContentItem && distFromBottom < 50 && distFromBottom > 0) {
            const speed = Math.max(3, (50 - distFromBottom) * 0.3);
            content.scrollTop += speed;
            this._showScrollHint('down');
          } else if (!isOverContentItem && distFromTop < 45 && distFromTop > 0) {
            const speed = Math.max(3, (45 - distFromTop) * 0.3);
            content.scrollTop -= speed;
            this._showScrollHint('up');
          } else {
            this._hideScrollHint();
          }
        } else {
          this._hideScrollHint();
        }
      } else {
        this._hideScrollHint();
      }
    } else {
      this.isPointerOverPanel = false;
      this.isPointerOverPanelHeader = false;
      this._hideScrollHint();
    }

    // Interactive button hover styling
    const isOverInteractive = hoveredEl && (hoveredEl.closest('button, a, .pset-header, .properties-close, .panel-action-btn, .prop-scroll-btn, .top-scroll-btn, .top-nav-btn, .properties-header') !== null);
    if (isOverInteractive) {
      this.cursorDot.style.filter = 'drop-shadow(0 0 6px #ffeb3b)';
    } else {
      this.cursorDot.style.filter = 'none';
    }

    // Finger hover effect on the full title bar (pick up / drop trigger zone)
    const headerEl = this.propertiesPanel?.querySelector('.properties-header');
    const isOverHeader = hoveredEl && headerEl && (hoveredEl === headerEl || headerEl.contains(hoveredEl)) && !hoveredEl.closest('.properties-close');
    if (isOverHeader && !this.panelAttachedToFinger) {
      headerEl.classList.add('finger-header-hover');
    } else if (headerEl) {
      headerEl.classList.remove('finger-header-hover');
    }

    // Finger hover effect on top expand-all button
    const expandAllBtn = this.propertiesPanel?.querySelector('#prop-expand-all');
    if (!this.panelAttachedToFinger && hoveredEl && expandAllBtn && (hoveredEl === expandAllBtn || expandAllBtn.contains(hoveredEl))) {
      expandAllBtn.classList.add('finger-active');
    } else if (expandAllBtn) {
      expandAllBtn.classList.remove('finger-active');
    }

    // Finger hover effect on individual property set headers
    const hoveredPsetHeader = hoveredEl ? hoveredEl.closest('.pset-header') : null;
    const allPsetHeaders = this.propertiesPanel?.querySelectorAll('.pset-header');
    if (allPsetHeaders) {
      allPsetHeaders.forEach((hdr) => {
        if (hdr === hoveredPsetHeader) {
          hdr.classList.add('finger-active');
        } else {
          hdr.classList.remove('finger-active');
        }
      });
    }

    // Direct finger touch / hover scrolling on top scroll buttons
    const scrollUpBtn = this.propertiesPanel?.querySelector('#prop-scroll-up');
    const scrollDownBtn = this.propertiesPanel?.querySelector('#prop-scroll-down');
    const content = this.propertiesPanel?.querySelector('.properties-content');

    let isTouchingScrollBtn = false;
    if (!this.panelAttachedToFinger && hoveredEl && content) {
      if (hoveredEl === scrollUpBtn || scrollUpBtn?.contains(hoveredEl)) {
        isTouchingScrollBtn = true;
        scrollUpBtn.classList.add('finger-active');
        scrollDownBtn?.classList.remove('finger-active');
        content.scrollTop -= 6;
        this._showScrollHint('up');
      } else if (hoveredEl === scrollDownBtn || scrollDownBtn?.contains(hoveredEl)) {
        isTouchingScrollBtn = true;
        scrollDownBtn.classList.add('finger-active');
        scrollUpBtn?.classList.remove('finger-active');
        content.scrollTop += 6;
        this._showScrollHint('down');
      }
    }

    if (!isTouchingScrollBtn) {
      scrollUpBtn?.classList.remove('finger-active');
      scrollDownBtn?.classList.remove('finger-active');
    }

    // Dwell (one-finger hold to select) logic
    const now = performance.now();

    if (!this.dwellAnchorPos) {
      this.dwellAnchorPos = { x: screenX, y: screenY };
      this.dwellStartTime = now;
      this.dwellTriggered = false;
      this._updateDwellRing(0);
      return;
    }

    const dist = Math.hypot(screenX - this.dwellAnchorPos.x, screenY - this.dwellAnchorPos.y);

    if (dist <= this.dwellTolerance) {
      if (this.dwellTriggered) {
        // If holding finger on top scroll button, pulse repeated jumps
        const isScrollBtn = hoveredEl && (hoveredEl === scrollUpBtn || hoveredEl === scrollDownBtn || hoveredEl.closest('.prop-scroll-btn, .top-scroll-btn'));
        if (isScrollBtn && now - this.dwellStartTime >= this.dwellDuration + 400) {
          this.dwellStartTime = now - this.dwellDuration + 200;
          this.audioManager.playInteractionClickSound();
          content?.scrollBy({ top: hoveredEl === scrollUpBtn || scrollUpBtn?.contains(hoveredEl) ? -160 : 160, behavior: 'smooth' });
        } else {
          this._updateDwellRing(1);
        }
        return;
      }

      // If actively edge-scrolling the text content (and not on a button), suppress dwell click
      if (!isTouchingScrollBtn && this.isPointerOverPanel && !isOverInteractive) {
        if (this.scrollHintEl && this.scrollHintEl.style.display !== 'none') {
          this._resetDwell();
          return;
        }
      }

      const elapsed = now - this.dwellStartTime;
      const progress = Math.min(1, elapsed / this.dwellDuration);
      this._updateDwellRing(progress);

      if (progress >= 1) {
        this.dwellTriggered = true;
        this._triggerDwellSelect(screenX, screenY);
      }
    } else {
      if (dist > this.dwellTolerance + 15) {
        this.dwellTriggered = false;
      }
      this.dwellAnchorPos = { x: screenX, y: screenY };
      this.dwellStartTime = now;
      this._updateDwellRing(0);
    }
  }

  _showScrollHint(direction) {
    if (!this.propertiesPanel) return;
    if (!this.scrollHintEl || !this.propertiesPanel.contains(this.scrollHintEl)) {
      this.scrollHintEl = document.createElement('div');
      this.propertiesPanel.appendChild(this.scrollHintEl);
    }
    this.scrollHintEl.className = `scroll-edge-hint ${direction === 'down' ? 'bottom' : 'top'}`;
    this.scrollHintEl.innerText = direction === 'down' ? '▼ Scrolling Down' : '▲ Scrolling Up';
    this.scrollHintEl.style.display = 'block';
  }

  _hideScrollHint() {
    if (this.scrollHintEl) {
      this.scrollHintEl.style.display = 'none';
    }
  }

  _togglePanelFingerMove() {
    this.panelAttachedToFinger = !this.panelAttachedToFinger;
    const btn = this.propertiesPanel?.querySelector('#prop-move-btn');
    const header = this.propertiesPanel?.querySelector('.properties-header');
    if (this.panelAttachedToFinger) {
      if (btn) btn.innerHTML = '📌 Drop';
      this.propertiesPanel?.classList.add('panel-floating-active');
      if (header) header.setAttribute('title', 'Attached to finger — hold still anywhere to drop');
    } else {
      if (btn) btn.innerHTML = '✥ Move';
      this.propertiesPanel?.classList.remove('panel-floating-active');
      if (header) {
        header.classList.remove('finger-header-hover');
        header.setAttribute('title', 'Hold still anywhere on title bar to move / drop panel');
      }
    }
  }

  _updateDwellRing(progress) {
    if (!this.dwellCircle) return;
    const offset = this.cursorRingCircumference * (1 - Math.max(0, Math.min(1, progress)));
    this.dwellCircle.style.strokeDashoffset = `${offset}`;
    if (progress >= 1) {
      this.dwellCircle.style.stroke = '#39ff14'; // Bright green flash on completion
    } else if (progress > 0.6) {
      this.dwellCircle.style.stroke = '#7df9ff';
    } else {
      this.dwellCircle.style.stroke = '#00ffff';
    }
  }

  _resetDwell() {
    this.dwellAnchorPos = null;
    this.dwellStartTime = 0;
    this.dwellTriggered = false;
    this._updateDwellRing(0);
  }

  async _triggerDwellSelect(screenX, screenY) {
    this.cursorDot.style.transform = 'scale(1.4)';
    setTimeout(() => {
      if (this.cursorDot) this.cursorDot.style.transform = 'scale(1)';
    }, 180);

    // If panel is attached to finger, dwelling drops it!
    if (this.panelAttachedToFinger) {
      this._togglePanelFingerMove();
      this.audioManager.playInteractionClickSound();
      return;
    }

    await this._onSelect(screenX, screenY, false);
  }

  async _onSelect(x, y, isNormalized = true) {
    let clientX = isNormalized ? x * this.renderDiv.clientWidth : x;
    let clientY = isNormalized ? y * this.renderDiv.clientHeight : y;

    clientX = Math.max(0, Math.min(this.renderDiv.clientWidth, clientX));
    clientY = Math.max(0, Math.min(this.renderDiv.clientHeight, clientY));

    this._showClickEffect(clientX, clientY);
    this.audioManager.playInteractionClickSound();

    // 1. Check if clicking on UI interactive element
    const elementAtPoint = document.elementFromPoint(clientX, clientY);
    if (elementAtPoint) {
      // First check close button
      if (elementAtPoint.closest('.properties-close')) {
        this._deselectElement();
        return;
      }

      // Entire title bar acts as move / drop trigger
      if (elementAtPoint.closest('.properties-header')) {
        this._togglePanelFingerMove();
        return;
      }

      const btn = elementAtPoint.closest('button, a');
      if (btn) {
        btn.click();
        return;
      }
      if (elementAtPoint.closest('.properties-panel')) {
        const psetHeader = elementAtPoint.closest('.pset-header');
        if (psetHeader) {
          const body = psetHeader.nextElementSibling;
          const toggle = psetHeader.querySelector('.pset-toggle');
          if (body) {
            body.classList.toggle('hidden');
            if (toggle) toggle.textContent = body.classList.contains('hidden') ? '▸' : '▾';
          }
          this._updateExpandAllButtonState();
        }
        return;
      }
    }

    // 2. If no IFC model is loaded, nothing 3D to pick
    if (!this.currentFragmentsModel || !this.currentModel) return;

    // 3. Raycast 3D elements
    const hitLocalId = await this._raycastElement(clientX, clientY);
    if (hitLocalId !== null && hitLocalId !== undefined) {
      await this._selectElement(hitLocalId);
    } else {
      this._deselectElement();
    }
  }

  _showClickEffect(clientX, clientY) {
    const ripple = document.createElement('div');
    ripple.className = 'click-ripple';
    ripple.style.left = `${clientX}px`;
    ripple.style.top = `${clientY}px`;
    this.renderDiv.appendChild(ripple);
    setTimeout(() => {
      ripple.remove();
    }, 380);
  }

  async _raycastElement(clientX, clientY) {
    if (!this.currentFragmentsModel || !this.currentModel) return null;

    // Strategy 1: ThatOpen Fragments raycast
    try {
      const mouse = new THREE.Vector2(clientX, clientY);
      const res = await this.currentFragmentsModel.raycast({
        camera: this.camera,
        mouse,
        dom: this.renderer.domElement,
      });
      if (res && res.localId !== undefined && res.localId !== null) {
        if (this.sectionEnabled && this.sectionPlane && res.point) {
          if (this.sectionPlane.distanceToPoint(res.point) >= 0) {
            return res.localId;
          }
        } else {
          return res.localId;
        }
      }
    } catch (e) {
      // Fragments internal raycast failed; fall through to Three.js raycasting
    }

    // Strategy 2: Three.js Raycaster fallback
    try {
      const rect = this.renderDiv.getBoundingClientRect();
      const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
      const ndcY = -((clientY - rect.top) / rect.height) * 2 + 1;
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);

      const hits = raycaster.intersectObjects(this.currentModel.children, true);
      for (const hit of hits) {
        if (this.sectionEnabled && this.sectionPlane) {
          if (this.sectionPlane.distanceToPoint(hit.point) < 0) {
            continue; // Clipped by section plane
          }
        }

        if (hit.object?.userData?.localId !== undefined) {
          return hit.object.userData.localId;
        }
        if (hit.object?.userData?.itemId !== undefined) {
          const localIds = await this.currentFragmentsModel.getLocalIdsFromItemIds([hit.object.userData.itemId]);
          if (localIds && localIds.length > 0 && localIds[0] !== undefined) {
            return localIds[0];
          }
        }
        if (hit.object?.userData?.itemIds?.size === 1) {
          const singleItemId = hit.object.userData.itemIds.values().next().value;
          const localIds = await this.currentFragmentsModel.getLocalIdsFromItemIds([singleItemId]);
          if (localIds && localIds.length > 0 && localIds[0] !== undefined) {
            return localIds[0];
          }
        }

        const geom = hit.object?.geometry;
        if (geom && geom.attributes && geom.attributes.id && hit.faceIndex !== undefined) {
          const idAttr = geom.attributes.id;
          const vIdx = geom.index ? geom.index.getX(hit.faceIndex * 3) : hit.faceIndex * 3;
          const itemId = idAttr.getX(vIdx);
          if (itemId !== undefined && itemId !== 0) {
            let localIds = await this.currentFragmentsModel.getLocalIdsFromItemIds([itemId]);
            if (localIds && localIds.length > 0 && localIds[0] !== undefined) {
              return localIds[0];
            }
            if (itemId > 0) {
              localIds = await this.currentFragmentsModel.getLocalIdsFromItemIds([itemId - 1]);
              if (localIds && localIds.length > 0 && localIds[0] !== undefined) {
                return localIds[0];
              }
            }
          }
        }
      }
    } catch (e) {
      console.warn('Raycast fallback error:', e);
    }

    return null;
  }

  async _selectElement(localId) {
    if (this.selectedLocalId !== null && this.selectedLocalId !== localId) {
      try {
        await this.currentFragmentsModel.resetColor([this.selectedLocalId]);
      } catch (e) {
        console.warn('Error resetting previous element color:', e);
      }
    }

    this.selectedLocalId = localId;

    try {
      await this.currentFragmentsModel.setColor([localId], new THREE.Color(0x00ffff));
    } catch (e) {
      console.warn('Error highlighting element:', e);
    }

    await this._updateSelectionHighlightBox(localId);
    this._showPropertiesLoading(localId);

    const props = await this._getElementProperties(localId);
    if (props && this.selectedLocalId === localId) {
      this._renderPropertiesPanel(props);
    }
  }

  _deselectElement() {
    this.panelAttachedToFinger = false;
    this._hideScrollHint();
    const header = this.propertiesPanel?.querySelector('.properties-header');
    if (header) header.classList.remove('finger-header-hover');
    if (this.selectedLocalId !== null && this.currentFragmentsModel) {
      try {
        this.currentFragmentsModel.resetColor([this.selectedLocalId]);
      } catch (e) {
        console.warn('Error resetting element color:', e);
      }
    }
    this.selectedLocalId = null;

    if (this.selectionBoxHelper) {
      this.scene.remove(this.selectionBoxHelper);
      this.selectionBoxHelper.geometry.dispose();
      this.selectionBoxHelper.material.dispose();
      this.selectionBoxHelper = null;
    }

    if (this.propertiesPanel) {
      this.propertiesPanel.classList.add('hidden');
      this.propertiesPanel.classList.remove('panel-floating-active');
      this.propertiesPanel.innerHTML = '';
    }
  }

  async _updateSelectionHighlightBox(localId) {
    if (!this.currentFragmentsModel || localId === null) {
      if (this.selectionBoxHelper) {
        this.scene.remove(this.selectionBoxHelper);
        this.selectionBoxHelper = null;
      }
      return;
    }
    try {
      const box = await this.currentFragmentsModel.getMergedBox([localId]);
      if (box && !box.isEmpty()) {
        if (!this.selectionBoxHelper) {
          this.selectionBoxHelper = new THREE.Box3Helper(box, new THREE.Color(0x00ffff));
          this.scene.add(this.selectionBoxHelper);
        } else {
          this.selectionBoxHelper.box.copy(box);
          this.selectionBoxHelper.visible = true;
        }
      } else if (this.selectionBoxHelper) {
        this.selectionBoxHelper.visible = false;
      }
    } catch (e) {
      console.warn('Failed to compute selection bounding box:', e);
      if (this.selectionBoxHelper) {
        this.selectionBoxHelper.visible = false;
      }
    }
  }

  async _getElementProperties(localId) {
    if (!this.currentFragmentsModel) return null;

    let category = 'Unknown Element';
    let guid = '';
    let name = '';
    const attributes = {};
    const propertySets = [];
    const quantities = [];

    try {
      const item = this.currentFragmentsModel.getItem(localId);
      if (item) {
        try {
          const cat = await item.getCategory();
          if (cat) category = cat;
        } catch (_) {}

        try {
          const g = await item.getGuid();
          if (g) guid = g;
        } catch (_) {}

        try {
          const rawAttrs = await item.getAttributes();
          if (rawAttrs) {
            rawAttrs.forEach((val, key) => {
              const unwrapped = this._unwrapPropertyValue(val);
              if (key.toLowerCase() === 'name' && unwrapped) {
                name = unwrapped;
              }
              attributes[key] = unwrapped;
            });
          }
        } catch (_) {}
      }
    } catch (_) {}

    try {
      const dataArr = await this.currentFragmentsModel.getItemsData([localId], {
        attributesDefault: true,
        relationsDefault: { attributes: true, relations: true },
      });

      if (dataArr && dataArr.length > 0) {
        const itemData = dataArr[0];
        if (!name && itemData.Name) {
          name = this._unwrapPropertyValue(itemData.Name);
        }
        if (!guid && itemData.GlobalId) {
          guid = this._unwrapPropertyValue(itemData.GlobalId);
        }

        const rels = itemData.IsDefinedBy || itemData.isDefinedBy || [];
        if (Array.isArray(rels)) {
          for (const relItem of rels) {
            this._processRelationItem(relItem, propertySets, quantities);
          }
        }

        for (const [key, val] of Object.entries(itemData)) {
          if (key === 'IsDefinedBy' || key === 'isDefinedBy' || key === 'localId') continue;
          if (key.startsWith('Pset_') || key.startsWith('Qto_')) {
            this._processRelationItem({ Name: key, ...val }, propertySets, quantities);
          }
        }
      }
    } catch (e) {
      console.warn('Failed to get deep items data:', e);
    }

    if (!name) {
      name = attributes.Name || attributes.name || `${category} #${localId}`;
    }

    return {
      localId,
      category,
      guid,
      name,
      attributes,
      propertySets,
      quantities,
    };
  }

  _processRelationItem(relItem, propertySets, quantities) {
    if (!relItem || typeof relItem !== 'object') return;

    const psetDef = relItem.RelatingPropertyDefinition || relItem;
    const psetName = this._unwrapPropertyValue(psetDef.Name) || 'Property Set';

    if (psetDef.Quantities && Array.isArray(psetDef.Quantities)) {
      const qList = [];
      for (const q of psetDef.Quantities) {
        const qName = this._unwrapPropertyValue(q.Name);
        let qVal = '';
        for (const [k, v] of Object.entries(q)) {
          if (k.toLowerCase().includes('value')) {
            qVal = this._unwrapPropertyValue(v);
            break;
          }
        }
        if (!qVal) qVal = this._unwrapPropertyValue(q.NominalValue) || '';
        if (qName) {
          qList.push({ name: qName, value: qVal || '—' });
        }
      }
      if (qList.length > 0) {
        quantities.push({ name: psetName, items: qList });
        return;
      }
    }

    if (psetDef.HasProperties && Array.isArray(psetDef.HasProperties)) {
      const propList = [];
      for (const p of psetDef.HasProperties) {
        const propName = this._unwrapPropertyValue(p.Name);
        const propVal = this._unwrapPropertyValue(p.NominalValue ?? p.nominalValue ?? p.Value ?? p.value);
        if (propName) {
          propList.push({ name: propName, value: propVal || '—' });
        }
      }
      if (propList.length > 0) {
        propertySets.push({ name: psetName, properties: propList });
        return;
      }
    }

    const entries = Object.entries(psetDef).filter(
      ([k]) => !['type', 'localId', 'GlobalId', 'Name', 'Description', 'RelatingPropertyDefinition'].includes(k)
    );
    if (entries.length > 0) {
      const propList = entries.map(([k, v]) => ({
        name: k,
        value: this._unwrapPropertyValue(v) || '—',
      }));
      if (psetName.startsWith('Qto_')) {
        quantities.push({ name: psetName, items: propList });
      } else {
        propertySets.push({ name: psetName, properties: propList });
      }
    }
  }

  _unwrapPropertyValue(val) {
    if (val === null || val === undefined) return '';
    if (typeof val === 'object') {
      if ('value' in val) return this._unwrapPropertyValue(val.value);
      if (Array.isArray(val)) return val.map((x) => this._unwrapPropertyValue(x)).join(', ');
      const keys = Object.keys(val);
      if (keys.length === 1 && typeof val[keys[0]] !== 'object') {
        return String(val[keys[0]]);
      }
      return '';
    }
    if (typeof val === 'boolean') return val ? 'True' : 'False';
    if (typeof val === 'number') {
      return Number.isInteger(val) ? String(val) : String(Math.round(val * 1000) / 1000);
    }
    return String(val);
  }

  _showPropertiesLoading(localId) {
    if (!this.propertiesPanel) return;
    this.propertiesPanel.innerHTML = `
      <div class="properties-header" title="Hold still anywhere on title bar to move / drop panel">
        <div class="properties-drag-handle">⋮⋮</div>
        <div class="properties-title-group">
          <span class="properties-badge">Loading...</span>
          <div class="properties-name">Element #${localId}</div>
        </div>
        <div class="properties-header-actions">
          <button type="button" class="panel-action-btn" id="prop-move-btn" title="Click or dwell to move panel with finger or mouse">${this.panelAttachedToFinger ? '📌 Drop' : '✥ Move'}</button>
          <button type="button" class="properties-close" title="Deselect element (Esc)">✕</button>
        </div>
      </div>
      <div class="properties-content" style="align-items: center; justify-content: center; padding: 30px 10px; color: #94a3b8;">
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
          <div style="width: 24px; height: 24px; border: 3px solid rgba(0, 191, 255, 0.2); border-top-color: #00bfff; border-radius: 50%; animation: spin 0.8s linear infinite;"></div>
          <span>Loading IFC properties...</span>
        </div>
      </div>
    `;
    this.propertiesPanel.classList.remove('hidden');

    const header = this.propertiesPanel.querySelector('.properties-header');
    if (header) {
      header.addEventListener('click', (e) => {
        if (e.target.closest('.properties-close')) return;
        this._togglePanelFingerMove();
      });
    }

    const closeBtn = this.propertiesPanel.querySelector('.properties-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this._deselectElement();
      });
    }
  }

  _renderPropertiesPanel(props) {
    if (!this.propertiesPanel) return;

    this.propertiesPanel.innerHTML = `
      <div class="properties-header" title="Hold still anywhere on title bar to move / drop panel">
        <div class="properties-drag-handle">⋮⋮</div>
        <div class="properties-title-group">
          <span class="properties-badge">${this._escapeHtml(props.category)}</span>
          <div class="properties-name" title="${this._escapeHtml(props.name)}">${this._escapeHtml(props.name)}</div>
        </div>
        <div class="properties-header-actions">
          <button type="button" class="panel-action-btn" id="prop-move-btn" title="Click or dwell to move panel with finger or mouse">${this.panelAttachedToFinger ? '📌 Drop' : '✥ Move'}</button>
          <button type="button" class="properties-close" title="Deselect element (Esc)">✕</button>
        </div>
      </div>
      <div class="properties-meta">
        <span><strong>ID:</strong> #${props.localId}</span>
        ${props.guid ? `<span title="GlobalId: ${this._escapeHtml(props.guid)}"><strong>GUID:</strong> ${this._escapeHtml(props.guid.slice(0, 14))}…</span>` : ''}
      </div>
      <div class="properties-top-nav">
        <button type="button" class="top-nav-btn" id="prop-expand-all" title="Expand or collapse all subpanels (touch or click)">⊞ Expand All</button>
        <button type="button" class="top-scroll-btn" id="prop-scroll-up" title="Scroll up (touch/point finger or click)">▲ Scroll Up</button>
        <button type="button" class="top-scroll-btn" id="prop-scroll-down" title="Scroll down (touch/point finger or click)">▼ Scroll Down</button>
      </div>
      <div class="properties-content">
        ${this._buildAttributesSectionHtml(props.attributes)}
        ${this._buildPropertySetsSectionHtml(props.propertySets)}
        ${this._buildQuantitiesSectionHtml(props.quantities)}
      </div>
    `;

    this.propertiesPanel.classList.remove('hidden');

    const header = this.propertiesPanel.querySelector('.properties-header');
    if (header) {
      header.addEventListener('click', (e) => {
        if (e.target.closest('.properties-close')) return;
        this._togglePanelFingerMove();
      });
    }

    const closeBtn = this.propertiesPanel.querySelector('.properties-close');
    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this._deselectElement();
      });
    }

    const expandAllBtn = this.propertiesPanel.querySelector('#prop-expand-all');
    if (expandAllBtn) {
      expandAllBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this._toggleAllPropertySets();
      });
    }

    const content = this.propertiesPanel.querySelector('.properties-content');
    const scrollUpBtn = this.propertiesPanel.querySelector('#prop-scroll-up');
    const scrollDownBtn = this.propertiesPanel.querySelector('#prop-scroll-down');

    if (scrollUpBtn && content) {
      scrollUpBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        content.scrollBy({ top: -160, behavior: 'smooth' });
      });
    }

    if (scrollDownBtn && content) {
      scrollDownBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        content.scrollBy({ top: 160, behavior: 'smooth' });
      });
    }

    const psetHeaders = this.propertiesPanel.querySelectorAll('.pset-header');
    psetHeaders.forEach((hdr) => {
      hdr.addEventListener('click', (e) => {
        e.stopPropagation();
        const body = hdr.nextElementSibling;
        const icon = hdr.querySelector('.pset-toggle');
        if (body) {
          body.classList.toggle('hidden');
          if (icon) {
            icon.textContent = body.classList.contains('hidden') ? '▸' : '▾';
          }
        }
        this._updateExpandAllButtonState();
      });
    });

    this._updateExpandAllButtonState();
  }

  _toggleAllPropertySets() {
    if (!this.propertiesPanel) return;
    const psetBodies = this.propertiesPanel.querySelectorAll('.pset-body');
    const psetToggles = this.propertiesPanel.querySelectorAll('.pset-toggle');
    const expandBtn = this.propertiesPanel.querySelector('#prop-expand-all');

    // If any body is hidden, expand all; otherwise collapse all
    const hasHidden = Array.from(psetBodies).some((b) => b.classList.contains('hidden'));
    const shouldExpand = hasHidden;

    psetBodies.forEach((body) => {
      if (shouldExpand) {
        body.classList.remove('hidden');
      } else {
        body.classList.add('hidden');
      }
    });

    psetToggles.forEach((toggle) => {
      toggle.textContent = shouldExpand ? '▾' : '▸';
    });

    if (expandBtn) {
      expandBtn.innerHTML = shouldExpand ? '⊟ Collapse All' : '⊞ Expand All';
      if (shouldExpand) {
        expandBtn.classList.add('all-expanded');
      } else {
        expandBtn.classList.remove('all-expanded');
      }
    }

    this.audioManager.playInteractionClickSound();
  }

  _updateExpandAllButtonState() {
    if (!this.propertiesPanel) return;
    const psetBodies = this.propertiesPanel.querySelectorAll('.pset-body');
    const expandBtn = this.propertiesPanel.querySelector('#prop-expand-all');
    if (!expandBtn || psetBodies.length === 0) return;

    const allOpen = Array.from(psetBodies).every((b) => !b.classList.contains('hidden'));
    if (allOpen) {
      expandBtn.innerHTML = '⊟ Collapse All';
      expandBtn.classList.add('all-expanded');
    } else {
      expandBtn.innerHTML = '⊞ Expand All';
      expandBtn.classList.remove('all-expanded');
    }
  }

  _buildAttributesSectionHtml(attributes) {
    const entries = Object.entries(attributes || {}).filter(
      ([key, val]) => val && !['type', 'localId'].includes(key)
    );
    if (entries.length === 0) return '';

    const rows = entries
      .map(
        ([key, val]) => `
          <tr>
            <td class="prop-key">${this._escapeHtml(key)}</td>
            <td class="prop-val">${this._escapeHtml(val)}</td>
          </tr>
        `
      )
      .join('');

    return `
      <div class="properties-section">
        <div class="properties-section-title">
          <span>📋</span> Attributes
        </div>
        <table class="properties-table">
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
  }

  _buildPropertySetsSectionHtml(propertySets) {
    if (!propertySets || propertySets.length === 0) return '';

    const cards = propertySets
      .map((pset, idx) => {
        const rows = (pset.properties || [])
          .map(
            (p) => `
            <tr>
              <td class="prop-key">${this._escapeHtml(p.name)}</td>
              <td class="prop-val">${this._escapeHtml(p.value)}</td>
            </tr>
          `
          )
          .join('');

        return `
          <div class="pset-card">
            <div class="pset-header">
              <span>${this._escapeHtml(pset.name)}</span>
              <span class="pset-toggle">${idx === 0 ? '▾' : '▸'}</span>
            </div>
            <div class="pset-body ${idx === 0 ? '' : 'hidden'}">
              <table class="properties-table">
                <tbody>${rows}</tbody>
              </table>
            </div>
          </div>
        `;
      })
      .join('');

    return `
      <div class="properties-section">
        <div class="properties-section-title">
          <span>🏷️</span> Property Sets (${propertySets.length})
        </div>
        ${cards}
      </div>
    `;
  }

  _buildQuantitiesSectionHtml(quantities) {
    if (!quantities || quantities.length === 0) return '';

    const cards = quantities
      .map((qto) => {
        const rows = (qto.items || [])
          .map(
            (q) => `
            <tr>
              <td class="prop-key">${this._escapeHtml(q.name)}</td>
              <td class="prop-val">${this._escapeHtml(q.value)}</td>
            </tr>
          `
          )
          .join('');

        return `
          <div class="pset-card">
            <div class="pset-header">
              <span>${this._escapeHtml(qto.name)}</span>
              <span class="pset-toggle">▾</span>
            </div>
            <div class="pset-body">
              <table class="properties-table">
                <tbody>${rows}</tbody>
              </table>
            </div>
          </div>
        `;
      })
      .join('');

    return `
      <div class="properties-section">
        <div class="properties-section-title">
          <span>📐</span> Quantities (${quantities.length})
        </div>
        ${cards}
      </div>
    `;
  }

  _escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
  }

  _onSectionStart() {
    this.sectionEnabled = true;
    this._updateSectionClipping();
    this.sectionBadge.style.backgroundColor = '#FFA500';
    this.sectionBadge.style.color = '#000000';
  }

  _onSectionMove(delta) {
    this.sectionOffset -= delta * this.renderDiv.clientHeight * SECTION_SENSITIVITY;
    this._rebuildSectionPlane();
  }

  _onSectionEnd() {
    this.sectionBadge.style.backgroundColor = 'white';
    this.sectionBadge.style.color = '#000000';
    // Section plane intentionally stays where it was left — no auto reset.
  }

  _updateSectionClipping() {
    if (!this.currentFragmentsModel?.object) return;
    this.currentFragmentsModel.object.traverse((child) => {
      if (!child.isMesh || !child.material) return;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        material.clippingPlanes = this.sectionEnabled ? [this.sectionPlane] : [];
        material.needsUpdate = true;
      });
    });
  }

  _onKeyDown(event) {
    if (event.key === ' ') {
      event.preventDefault();
      this._toggleGesturePause();
    } else if (event.key.toLowerCase() === 'r') this._resetView();
    else if (event.key.toLowerCase() === 'b') this._toggleBackground();
    else if (event.key.toLowerCase() === 'l') this._toggleLandmarkOverlay();
    else if (event.key.toLowerCase() === 'i') this._toggleInfoPanel();
    else if (event.key === 'Escape') {
      if (this.selectedLocalId !== null) {
        this._deselectElement();
      } else if (this.infoPanelOpen) {
        this._closeInfoPanel();
      }
    } else if (event.key === '1') this._setSectionOrientation(0);
    else if (event.key === '2') this._setSectionOrientation(1);
    else if (event.key === '3') this._setSectionOrientation(2);
  }

  _setSectionOrientation(index) {
    this.sectionOrientationIndex = index;
    this._rebuildSectionPlane();
    this._updateSectionBadgeText();
  }

  _rebuildSectionPlane() {
    const localNormal = SECTION_ORIENTATIONS[this.sectionOrientationIndex].normal;
    const modelPosition = this.currentModel?.position ?? new THREE.Vector3(0, 0, this.modelDefaultZ);
    const modelRotation = this.currentModel?.quaternion ?? new THREE.Quaternion();
    const normal = localNormal.clone().applyQuaternion(modelRotation).normalize();
    const planePoint = modelPosition.clone().addScaledVector(normal, this.sectionOffset);
    this.sectionPlane.setFromNormalAndCoplanarPoint(normal, planePoint);

    if (this.sectionPlaneVisual) {
      this.sectionPlaneVisual.position.copy(planePoint);
      this.sectionPlaneVisual.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal);
    }
  }

  _updateSectionBadgeText() {
    if (!this.sectionBadge) return;
    this.sectionBadge.innerText = `Section: ${SECTION_ORIENTATIONS[this.sectionOrientationIndex].label} (1/2/3)`;
  }

  // ---------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------

  _animate() {
    requestAnimationFrame(this._animate.bind(this));
    this._updateGestureBadge();
    this._renderBackgroundBlur();
    this.renderer.render(this.scene, this.camera);
  }

  _updateGestureBadge() {
    if (!this.handControl) return;
    if (this.handControl.isPaused) {
      this.gestureBadge.innerText = 'Gestures paused';
      this.gestureBadge.style.backgroundColor = '#d9d9d9';
      this.pauseButton.innerText = 'Resume gestures';
      this.cursorDot.style.display = 'none';
      this._resetDwell();
      return;
    }
    const state = this.handControl.state;
    this.pauseButton.innerText = 'Pause gestures';
    if (this._lastBadgeState === state) return;
    this._lastBadgeState = state;
    this.gestureBadge.innerText = GESTURE_STATE_LABELS[state] || state;
    this.gestureBadge.title = `${state}: ${this._gestureHint(state)}`;
    this.gestureBadge.style.backgroundColor = GESTURE_STATE_COLORS[state] || '#888888';
    if (state !== GestureState.POINTING && state !== GestureState.HAND_VISIBLE) {
      this.cursorDot.style.display = 'none';
      this._resetDwell();
    }
  }

  _gestureHint(state) {
    return {
      [GestureState.HAND_VISIBLE]: 'Point with one finger to aim and select',
      [GestureState.POINTING]: 'Hold finger still on an element or button to select (1s)',
      [GestureState.PINCHING]: 'Move while pinching to rotate',
      [GestureState.ROTATING]: 'Release pinch to stop rotating',
      [GestureState.PANNING]: 'Move your open palm to pan',
      [GestureState.TWO_HAND_ZOOM]: 'Move hands apart or together to zoom',
      [GestureState.SECTION_ACTIVE]: 'Move or twist two raised fingers for section cut',
    }[state] || 'Show a hand to begin';
  }

  _toggleGesturePause() {
    if (!this.handControl) return;
    if (this.handControl.isPaused) {
      this.handControl.resume();
    } else {
      this.handControl.pause();
      this._resetDwell();
    }
    this._lastBadgeState = null;
    this._updateGestureBadge();
  }

  _resetView() {
    if (this.currentModel) {
      this.currentModel.position.set(0, 0, this.modelDefaultZ);
      this.currentModel.rotation.set(0, 0, 0);
      this.currentModel.scale.set(this.baseModelScale, this.baseModelScale, this.baseModelScale);
    }
    this.sectionOffset = 0;
    this.sectionEnabled = false;
    this._rebuildSectionPlane();
    this._updateSectionClipping();
    if (this.selectedLocalId !== null) {
      this._updateSelectionHighlightBox(this.selectedLocalId);
    }
    this.handControl?.reset();
    this._resetDwell();
    this._drawLandmarks([]);
    this._lastBadgeState = null;
    this.cursorDot.style.display = 'none';
    this._updateSectionBadgeText();
    this._updateGestureBadge();
  }

  _toggleBackground() {
    if (!this.backgroundVisible) {
      this.backgroundVisible = true;
      this.backgroundBlurred = false;
    } else if (!this.backgroundBlurred) {
      this.backgroundBlurred = true;
    } else {
      this.backgroundVisible = false;
      this.backgroundBlurred = false;
    }
    if (this.backgroundBlurred && !this.backgroundSegmenter) {
      this._setupBackgroundSegmenter();
    }
    this.videoElement.style.opacity = this.backgroundVisible ? '1' : '0';
    this.videoElement.style.filter = this.backgroundBlurred && !this.backgroundSegmenter
      ? `blur(${BACKGROUND_BLUR_PX}px)` : 'none';
    this.backgroundCanvas.style.display = this.backgroundBlurred ? 'block' : 'none';
    this._updateBackgroundButton();
  }

  _renderBackgroundBlur() {
    if (!this.backgroundBlurred || !this.backgroundSegmenter || !this.videoElement.videoWidth) return;
    const width = this.renderDiv.clientWidth;
    const height = this.renderDiv.clientHeight;
    if (this.backgroundCanvas.width !== width || this.backgroundCanvas.height !== height) {
      this.backgroundCanvas.width = width;
      this.backgroundCanvas.height = height;
      this.backgroundSourceCanvas.width = width;
      this.backgroundSourceCanvas.height = height;
    }
    const sourceContext = this.backgroundSourceCanvas.getContext('2d');
    const context = this.backgroundCanvas.getContext('2d');
    if (!sourceContext || !context) return;

    const crop = this._videoCropRect(width, height);
    sourceContext.clearRect(0, 0, width, height);
    this._drawMirroredVideo(sourceContext, width, height, crop);
    let result;
    try {
      result = this.backgroundSegmenter.segmentForVideo(this.videoElement, performance.now());
    } catch (error) {
      console.warn('Background blur frame failed; showing clear camera view.', error);
      this.backgroundBlurred = false;
      this.backgroundCanvas.style.display = 'none';
      this.videoElement.style.opacity = this.backgroundVisible ? '1' : '0';
      this.videoElement.style.filter = this.backgroundBlurred && !this.backgroundSegmenter
        ? `blur(${BACKGROUND_BLUR_PX}px)` : 'none';
      this._updateBackgroundButton();
      return;
    }
    const mask = result.categoryMask;
    if (!mask) return;
    const maskWidth = mask.width;
    const maskHeight = mask.height;
    if (this.backgroundMaskCanvas.width !== maskWidth || this.backgroundMaskCanvas.height !== maskHeight) {
      this.backgroundMaskCanvas.width = maskWidth;
      this.backgroundMaskCanvas.height = maskHeight;
    }
    const maskContext = this.backgroundMaskCanvas.getContext('2d');
    if (!maskContext) return;
    const values = mask.getAsUint8Array();
    const maskPixels = maskContext.createImageData(maskWidth, maskHeight);
    for (let index = 0; index < values.length; index++) {
      // selfie_segmenter's category mask is 0 = person/foreground, nonzero =
      // background — this alpha marks where to cut the "stay sharp" hole.
      const alpha = values[index] === 0 ? 255 : 0;
      const pixelIndex = index * 4;
      maskPixels.data[pixelIndex] = 255;
      maskPixels.data[pixelIndex + 1] = 255;
      maskPixels.data[pixelIndex + 2] = 255;
      maskPixels.data[pixelIndex + 3] = alpha;
    }
    maskContext.putImageData(maskPixels, 0, 0);

    context.clearRect(0, 0, width, height);
    context.filter = `blur(${BACKGROUND_BLUR_PX}px)`;
    this._drawMirroredVideo(context, width, height, crop);
    context.filter = 'none';
    context.globalCompositeOperation = 'destination-out';
    context.save();
    context.translate(width, 0);
    context.scale(-1, 1);
    context.drawImage(this.backgroundMaskCanvas, width - crop.x - crop.width, crop.y, crop.width, crop.height);
    context.restore();
    context.globalCompositeOperation = 'destination-over';
    context.drawImage(this.backgroundSourceCanvas, 0, 0, width, height);
    context.globalCompositeOperation = 'source-over';
    mask.close?.();
  }

  _videoCropRect(width, height) {
    const scale = Math.max(width / this.videoElement.videoWidth, height / this.videoElement.videoHeight);
    return {
      x: (width - this.videoElement.videoWidth * scale) / 2,
      y: (height - this.videoElement.videoHeight * scale) / 2,
      width: this.videoElement.videoWidth * scale,
      height: this.videoElement.videoHeight * scale,
    };
  }

  _drawMirroredVideo(context, width, height, crop) {
    context.save();
    context.translate(width, 0);
    context.scale(-1, 1);
    context.drawImage(this.videoElement, width - crop.x - crop.width, crop.y, crop.width, crop.height);
    context.restore();
  }

  _setupLandmarkOverlay() {
    this.landmarkOverlay = document.createElement('canvas');
    this.landmarkOverlay.className = 'landmark-overlay';
    this.landmarkOverlayVisible = true;
    this.renderDiv.appendChild(this.landmarkOverlay);
    this._resizeLandmarkOverlay();
  }

  _toggleLandmarkOverlay() {
    this.landmarkOverlayVisible = !this.landmarkOverlayVisible;
    this.landmarkOverlay.style.display = this.landmarkOverlayVisible ? 'block' : 'none';
    this._updateLandmarkButton();
  }

  _updateLandmarkButton() {
    if (!this.landmarkButton) return;
    this.landmarkButton.innerText = this.landmarkOverlayVisible ? 'Hide hands' : 'Show hands';
    this.landmarkButton.setAttribute('aria-pressed', String(this.landmarkOverlayVisible));
  }

  _drawLandmarks(hands) {
    const context = this.landmarkOverlay.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, this.landmarkOverlay.width, this.landmarkOverlay.height);
    if (!this.landmarkOverlayVisible) return;

    const connections = [
      [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
      [0, 13, 14, 15, 16], [0, 17, 18, 19, 20], [5, 9, 13, 17],
    ];
    context.lineWidth = 3;
    context.strokeStyle = '#00e5ff';
    context.fillStyle = '#ffffff';
    hands.forEach((landmarks) => {
      connections.forEach((chain) => {
        context.beginPath();
        chain.forEach((index, pointIndex) => {
          const point = landmarks[index];
          const { x, y } = this._landmarkToOverlay(point);
          if (pointIndex === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.stroke();
      });
      landmarks.forEach((point) => {
        const { x, y } = this._landmarkToOverlay(point);
        context.beginPath();
        context.arc(x, y, 4, 0, Math.PI * 2);
        context.fill();
      });
    });
  }

  _landmarkToOverlay(point) {
    const width = this.renderDiv.clientWidth || this.landmarkOverlay?.width || window.innerWidth;
    const height = this.renderDiv.clientHeight || this.landmarkOverlay?.height || window.innerHeight;
    const sourceWidth = this.videoElement?.videoWidth || width;
    const sourceHeight = this.videoElement?.videoHeight || height;
    const coverScale = Math.max(width / sourceWidth, height / sourceHeight);
    const renderedWidth = sourceWidth * coverScale;
    const renderedHeight = sourceHeight * coverScale;
    return {
      x: point.x * renderedWidth + (width - renderedWidth) / 2,
      y: point.y * renderedHeight + (height - renderedHeight) / 2,
    };
  }

  _resizeLandmarkOverlay() {
    if (!this.landmarkOverlay) return;
    this.landmarkOverlay.width = this.renderDiv.clientWidth;
    this.landmarkOverlay.height = this.renderDiv.clientHeight;
  }

  _updateBackgroundButton() {
    if (!this.backgroundButton) return;
    this.backgroundButton.innerText = this.backgroundSegmenterLoading
      ? 'Loading blur...'
      : !this.backgroundVisible
        ? 'Show background'
        : this.backgroundBlurred ? 'Blurred background' : 'Blur background';
    this.backgroundButton.setAttribute('aria-pressed', String(this.backgroundBlurred));
  }

  start() {
    this.renderDiv.addEventListener('click', () => {
      this.audioManager.resumeContext();
      if (this.gameState === 'error') {
        this._restartGame();
      }
    });
  }

  // ---------------------------------------------------------------------
  // Error / resize / restart
  // ---------------------------------------------------------------------

  _showError(message, showRetry = false, showSkipCamera = false) {
    this._showStatusScreen(`ERROR: ${message}`, '#ff9800');
    this.gameState = 'error';
    if (showRetry) {
      this._showRetryButton(showSkipCamera);
    }
  }

  _showRetryButton(showSkipCamera = false) {
    let existingActions = document.getElementById('init-actions-wrapper');
    if (existingActions) existingActions.remove();
    let oldBtn = document.getElementById('retry-init-btn');
    if (oldBtn) oldBtn.remove();

    const wrapper = document.createElement('div');
    wrapper.id = 'init-actions-wrapper';
    Object.assign(wrapper.style, {
      marginTop: '22px',
      display: 'flex',
      gap: '12px',
      justifyContent: 'center',
      flexWrap: 'wrap',
      pointerEvents: 'auto',
    });

    const retryBtn = document.createElement('button');
    retryBtn.type = 'button';
    retryBtn.innerText = '🔄 Retry Camera';
    Object.assign(retryBtn.style, {
      padding: '12px 22px',
      fontSize: '15px',
      fontWeight: 'bold',
      color: '#ffffff',
      backgroundColor: '#007bff',
      border: '2px solid #0056b3',
      borderRadius: '6px',
      cursor: 'pointer',
      boxShadow: '0 4px 6px rgba(0,0,0,0.4)',
    });
    retryBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      wrapper.remove();
      this._showStatusScreen('Connecting to camera...', 'white');
      await this._init();
    });
    wrapper.appendChild(retryBtn);

    if (showSkipCamera) {
      const skipBtn = document.createElement('button');
      skipBtn.type = 'button';
      skipBtn.innerText = '📁 Continue Without Camera (Mouse 3D Mode)';
      Object.assign(skipBtn.style, {
        padding: '12px 22px',
        fontSize: '15px',
        fontWeight: 'bold',
        color: '#ffffff',
        backgroundColor: '#28a745',
        border: '2px solid #1e7e34',
        borderRadius: '6px',
        cursor: 'pointer',
        boxShadow: '0 4px 6px rgba(0,0,0,0.4)',
      });
      skipBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        wrapper.remove();
        this._showStatusScreen('Starting 3D viewer...', 'white');
        await this._initWithoutCamera();
      });
      wrapper.appendChild(skipBtn);
    }

    this.statusContainer.appendChild(wrapper);
  }

  async _initWithoutCamera() {
    this._setupDOM();
    this._setupThree();
    this.videoElement.style.display = 'none';

    try {
      await this._setupIfc();
    } catch (ifcErr) {
      console.error('IFC setup failed:', ifcErr);
      this._showError(`IFC engine initialization failed: ${ifcErr.message || ifcErr}`, true, false);
      return;
    }

    if (this.gestureBadge) {
      this.gestureBadge.innerText = 'Mouse & Keys Mode';
      this.gestureBadge.style.backgroundColor = '#6c757d';
    }

    this.audioManager.resumeContext();
    this.clock.start();
    if (!this._eventsBound) {
      window.addEventListener('resize', this._onResize.bind(this));
      window.addEventListener('keydown', this._onKeyDown.bind(this));
      this._eventsBound = true;
    }

    this.gameState = 'tracking';
    this._updateStatusOverlayVisibility();
    if (!this._animating) {
      this._animating = true;
      this._animate();
    }
  }

  async _restartGame() {
    if (this.gameState === 'error') {
      this._showStatusScreen('Re-initializing...', 'white');
      await this._init();
    } else {
      this._updateStatusOverlayVisibility();
      this.gameState = 'tracking';
      this.clock.start();
    }
  }

  _onResize() {
    const width = this.renderDiv.clientWidth;
    const height = this.renderDiv.clientHeight;
    if (this.camera) {
      this.camera.left = width / -2;
      this.camera.right = width / 2;
      this.camera.top = height / 2;
      this.camera.bottom = height / -2;
      this.camera.updateProjectionMatrix();
    }
    if (this.renderer) {
      this.renderer.setSize(width, height);
    }
    if (this.videoElement) {
      this.videoElement.style.width = `${width}px`;
      this.videoElement.style.height = `${height}px`;
    }
    this._resizeLandmarkOverlay();
  }
}
