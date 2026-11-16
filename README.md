# HandyBIM

Drag and drop an IFC/BIM model over your webcam feed and inspect it entirely with touchless hand gestures — no mouse or keyboard required.

Built with [Three.js](https://threejs.org/), [That Open Components](https://github.com/ThatOpen/engine_components) (IFC loading via Fragments), and [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) for real-time hand tracking and background segmentation.

---

## Features

- **Drag & Drop IFC Loading** — drop any `.ifc` file straight onto the camera view to view and inspect.
- **Physical Fingertip Raycasting** — extended landmark tracking (+35% projection) places the aim reticle accurately on your physical fingernail tip.
- **One-Finger Dwell Click & Selection** — point at 3D building elements or UI buttons and hold still for 0.8s to trigger an animated circular progress ring, audio click, and selection. Pinching is cleanly preserved for 3D model rotation.
- **Interactive IFC Properties Inspector**:
  - Highlights selected building elements in cyan with a 3D bounding box (`Box3Helper`).
  - Displays element category, name, Express ID, and GlobalId (GUID).
  - Categorizes deep IFC data into collapsible **Attributes**, **Property Sets (Psets)**, and **Quantities (Qto)**.
  - **Whole Title Bar Move & Drop**: Point and dwell anywhere on the top title bar to attach the panel to your fingertip; it smoothly glides across the screen with your hand. Hold still anywhere to drop and lock it in place.
  - **Top Action Bar**:
    - **`[ ⊞ Expand All ]` / `[ ⊟ Collapse All ]`**: Single touch expands or collapses all subpanels (`Pset_ColumnCommon`, dimensions, etc.) at once.
    - **`[ ▲ Scroll Up ]` / `[ ▼ Scroll Down ]`**: Placed right at the top so your hand never has to reach down to the edge of the camera view.
- **Smooth Multi-Modal Scrolling**:
  - **Instant Hover-to-Scroll**: Simply hovering your fingertip over the top `▲`/`▼` buttons smoothly scrolls the list continuously without waiting.
  - **Continuous Edge Auto-Scrolling**: Pointing near the top or bottom of the panel content automatically scrolls with an animated indicator (`▲ Scrolling Up` / `▼ Scrolling Down`).
  - **Open-Palm Air-Swipe**: Show an open palm over the panel to scroll up or down.
  - **Mouse & Touchscreen Parity**: Full native mouse wheel scrolling, custom draggable scrollbar, and mouse drag on the header.
- **Live Section Cuts** — hold up two fingers (peace sign ✌️) to slice through the model; switch between horizontal and vertical planes with keys `1`, `2`, or `3`.
- **Webcam Background Controls** — cycle between clear camera, person-aware background blur (MediaPipe Selfie Segmenter), or hidden background.
- **Camera Fallback & Mouse Mode** — includes automatic camera error detection with one-click **"🔄 Retry Camera"** and **"📁 Continue Without Camera (Mouse 3D Mode)"** options.

---

## Controls

### Hand Gestures (Touchless)

| Gesture | Action |
| --- | --- |
| **Point (1 finger)** | Aim cursor at 3D elements or UI buttons |
| **Hold Still / Dwell (0.8s)** | Select 3D element, click button, or toggle title bar move/drop |
| **Pinch + Drag** | Rotate 3D BIM model (or drag panel header) |
| **Open Palm** | Pan 3D model in space / air-swipe scroll over properties panel |
| **Two Hands (move apart/together)** | Zoom in / out |
| **✌️ Two Fingers** | Interactive section cut plane |
| **Hover on Top `▲` / `▼` Buttons** | Smooth continuous auto-scrolling inside properties panel |
| **Dwell on Title Bar** | Attach panel to fingertip (green glow); dwell anywhere to drop |
| **Dwell on `⊞ Expand All`** | Expand or collapse all property sets & quantities |
| **Thumbs Up** | Pause / resume hand gestures |
| **Fist** | Reset view and camera orientation |

### Mouse & Keyboard Equivalents

| Input | Action |
| --- | --- |
| **Left Click + Drag** | Rotate 3D model / drag properties panel header |
| **Right Click + Drag** | Pan 3D model |
| **Mouse Wheel** | Zoom 3D model (or scroll naturally inside properties panel) |
| **Click Title Bar** | Pick up or drop properties panel |
| **Click Buttons** | `⊞ Expand All`, `▲ Scroll Up`, `▼ Scroll Down`, `✕ Close` |
| `Space` | Pause / resume gesture recognition |
| `R` | Reset view to default orientation and scale |
| `B` | Cycle webcam background (Clear ➔ Blurred ➔ Dark) |
| `L` | Toggle hand skeleton landmarks overlay |
| `I` | Toggle on-screen gestures legend |
| `1` / `2` / `3` | Switch section cut orientation (Horizontal, Vertical X, Vertical Z) |
| `Esc` | Deselect element and close properties panel |

---

## Getting Started

### Prerequisites

- Node.js (v18 or higher recommended)
- A webcam connected to your computer

### Installation

```bash
git clone https://github.com/nmuthusiva/HandyBIM.git
cd HandyBIM
npm install
```

### Development Server

```bash
npm run dev
```

Open the local development URL (`http://localhost:5173/`), allow camera access in your browser, and drag an `.ifc` file into the window.

### Production Build

```bash
npm run build
npm run preview
```

---

## Tech Stack

- **[Three.js](https://threejs.org/)** — 3D scene rendering, lighting, materials, and camera management.
- **[@thatopen/components](https://github.com/ThatOpen/engine_components) & [@thatopen/fragments](https://github.com/ThatOpen/engine_components)** — High-performance IFC loading, fragment generation, spatial data extraction, and geometry raycasting.
- **[@mediapipe/tasks-vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)** — Real-time 21-point 3D hand landmark detection and selfie segmentation.
- **[Vite](https://vitejs.dev/)** — Modern development server and production bundler.
