import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { computeRidgeGradientField, computeRidgePca, ridgeStepDetails } from "./ridgePca.js";

const GREEN = 0x10a37f;
const RED = 0xef4146;
const BLUE = 0x7892c6;
const TEXT = "#d7dfeb";
const AXIS_LABELS = [
  { key: "PC1", axis: 0 },
  { key: "PC2", axis: 1 },
  { key: "PC3", axis: 2 },
];

const fmt = (value, digits = 5) =>
  Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${value.toFixed(digits)}` : "n/a";

const short = (value, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : value === null || value === undefined ? "n/a" : String(value);

const pct = (value) => (Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a");

function liftColor(value) {
  if (!Number.isFinite(value)) return BLUE;
  return value >= 0 ? GREEN : RED;
}

function getRowsByStep(run) {
  const rows = new Map();
  for (const candidate of run?.candidates || []) {
    for (const row of candidate.rows || []) {
      rows.set(Number(row.index) + 1, {
        ...row,
        candidateId: candidate.id,
      });
    }
  }
  return rows;
}

function axisLabelText(label, loadings) {
  const features = (loadings || [])
    .slice(0, 2)
    .map((item) => item.feature)
    .join(" + ");
  return features ? `${label}: ${features}` : label;
}

function makeTextSprite(text, color = TEXT, options = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = options.width || 512;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = `${options.weight || 600} ${options.size || 30}px "Source Sans 3", sans-serif`;
  context.fillStyle = color;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, canvas.width / 2, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(options.scaleX || 1.35, options.scaleY || 0.26, 1);
  return sprite;
}

function gradientColor(strength) {
  return new THREE.Color(GREEN).lerp(new THREE.Color(RED), Math.min(1, Math.max(0, strength)));
}

function disposeScene(scene) {
  scene.traverse((object) => {
    if (object.geometry) object.geometry.dispose();
    if (object.material) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => {
        if (material.map) material.map.dispose();
        material.dispose();
      });
    }
  });
}

function RidgeScene({ pca, field, rowsByStep, addedRow, setHover }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !field?.vectors?.length) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05070a);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
    camera.position.set(6.4, 4.9, 7.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x05070a, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.className = "ridgeCanvas";
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.55;
    controls.panSpeed = 0.45;
    controls.minDistance = 3.2;
    controls.maxDistance = 18;

    scene.add(new THREE.AmbientLight(0xffffff, 0.64));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.16);
    keyLight.position.set(5, 8, 7);
    scene.add(keyLight);

    const scale = 3.55 / Math.max(field.bounds?.radius || 1e-9, 1e-9);
    const toScene = (coords) =>
      new THREE.Vector3(
        ((coords?.[0] || 0) - (field.bounds?.center?.[0] || 0)) * scale,
        ((coords?.[1] || 0) - (field.bounds?.center?.[1] || 0)) * scale,
        ((coords?.[2] || 0) - (field.bounds?.center?.[2] || 0)) * scale,
      );

    const grid = new THREE.GridHelper(8.2, 8, 0x1a2230, 0x101722);
    grid.material.transparent = true;
    grid.material.opacity = 0.34;
    scene.add(grid);

    const axisMaterial = new THREE.LineBasicMaterial({ color: 0x334052, transparent: true, opacity: 0.72 });
    AXIS_LABELS.map(({ key, axis }) => {
      const startCoords = [0, 0, 0];
      const endCoords = [0, 0, 0];
      startCoords[axis] = field.bounds.mins[axis];
      endCoords[axis] = field.bounds.maxs[axis];
      return [
        toScene(startCoords),
        toScene(endCoords),
        axisLabelText(key, pca.featureLoadings?.[axis]),
        toScene(endCoords).add(new THREE.Vector3(axis === 0 ? 0.28 : 0, axis === 1 ? 0.28 : 0, axis === 2 ? 0.28 : 0)),
      ];
    }).forEach(([start, end, label, position]) => {
      const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
      scene.add(new THREE.Line(geometry, axisMaterial.clone()));
      const sprite = makeTextSprite(label, "#b6c0cf", { size: 26, scaleX: 1.75, scaleY: 0.28 });
      sprite.position.copy(position);
      scene.add(sprite);
    });

    const trajectoryPoints = (field.trajectory || []).map(toScene);
    for (let index = 1; index < trajectoryPoints.length; index += 1) {
      const material = new THREE.LineBasicMaterial({
        color: index <= field.index ? 0x708095 : 0x2b3442,
        transparent: true,
        opacity: index <= field.index ? 0.54 : 0.22,
      });
      const geometry = new THREE.BufferGeometry().setFromPoints([trajectoryPoints[index - 1], trajectoryPoints[index]]);
      scene.add(new THREE.Line(geometry, material));
    }

    const pointMeshes = [];
    const pathGeometry = new THREE.SphereGeometry(0.052, 18, 12);
    const previousGeometry = new THREE.SphereGeometry(0.105, 24, 16);
    trajectoryPoints.forEach((point, index) => {
      const row = rowsByStep.get(index + 1);
      const isCurrent = index === field.index;
      const isPrevious = index === field.index - 1;
      if (!isPrevious && !isCurrent && index % Math.max(1, Math.floor(trajectoryPoints.length / 55)) !== 0) return;
      const mesh = new THREE.Mesh(
        isPrevious ? previousGeometry : pathGeometry,
        new THREE.MeshStandardMaterial({
          color: isPrevious ? 0xd8dee8 : liftColor(Number(row?.liftNll)),
          transparent: true,
          opacity: isPrevious ? 0.9 : 0.68,
          roughness: 0.62,
          metalness: 0.02,
        }),
      );
      mesh.position.copy(point);
      mesh.userData = {
        step: index + 1,
        lift: row?.liftNll,
        candidateId: row?.candidateId,
        kind: isPrevious ? "previous" : "trajectory",
      };
      scene.add(mesh);
      pointMeshes.push(mesh);
    });

    for (const vector of field.vectors) {
      const origin = toScene(vector.coords);
      const direction = new THREE.Vector3(
        vector.descent[0] * scale,
        vector.descent[1] * scale,
        vector.descent[2] * scale,
      );
      const magnitude = direction.length();
      if (magnitude <= 1e-9) continue;

      const strength = Math.min(1, vector.gradientNorm / Math.max(field.maxGradient, 1e-9));
      const length = 0.2 + 0.42 * Math.sqrt(strength);
      const arrow = new THREE.ArrowHelper(direction.normalize(), origin, length, gradientColor(strength), 0.13, 0.075);
      arrow.line.material.transparent = true;
      arrow.line.material.opacity = 0.36 + 0.38 * strength;
      arrow.cone.material.transparent = true;
      arrow.cone.material.opacity = 0.52 + 0.32 * strength;
      scene.add(arrow);
    }

    const currentColor = liftColor(Number(addedRow?.liftNll));
    const currentPosition = toScene(field.activeProjection);
    const currentMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 36, 24),
      new THREE.MeshStandardMaterial({
        color: currentColor,
        emissive: new THREE.Color(0x15201d),
        roughness: 0.54,
        metalness: 0.04,
      }),
    );
    currentMesh.position.copy(currentPosition);
    currentMesh.userData = {
      step: field.index + 1,
      lift: addedRow?.liftNll,
      candidateId: addedRow?.candidateId,
      kind: "current",
      gradientNorm: field.currentGradientNorm,
      loss: field.currentLoss,
    };
    scene.add(currentMesh);
    pointMeshes.push(currentMesh);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.32, 0.008, 8, 96),
      new THREE.MeshBasicMaterial({ color: 0xe4eaf3, transparent: true, opacity: 0.76 }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(currentPosition);
    scene.add(ring);

    const currentLabel = makeTextSprite(`current t ${field.index + 1}`, "#e4eaf3");
    currentLabel.position.copy(currentPosition).add(new THREE.Vector3(0.24, 0.42, 0));
    scene.add(currentLabel);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const handlePointerMove = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const [hit] = raycaster.intersectObjects(pointMeshes, false);
      if (!hit) {
        setHover(null);
        return;
      }
      setHover({
        ...hit.object.userData,
        x: event.clientX - rect.left + 12,
        y: event.clientY - rect.top + 12,
      });
    };
    const handlePointerLeave = () => setHover(null);
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerleave", handlePointerLeave);

    const resize = () => {
      const width = Math.max(320, mount.clientWidth);
      const height = Math.max(320, mount.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    resize();

    let frame = 0;
    let raf = 0;
    const animate = () => {
      frame += 1;
      const pulse = 1 + Math.sin(frame * 0.055) * 0.08;
      currentMesh.scale.setScalar(pulse);
      ring.rotation.z += 0.004;
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointermove", handlePointerMove);
      renderer.domElement.removeEventListener("pointerleave", handlePointerLeave);
      controls.dispose();
      if (renderer.domElement.parentElement === mount) {
        mount.removeChild(renderer.domElement);
      }
      disposeScene(scene);
      renderer.dispose();
    };
  }, [addedRow, field, pca, rowsByStep, setHover]);

  return <div className="ridgeViewport" ref={mountRef} />;
}

export default function RidgePcaView({ run, step }) {
  const [hover, setHover] = useState(null);
  const [readoutOpen, setReadoutOpen] = useState(false);
  const pca = useMemo(() => computeRidgePca(run), [run]);
  const rowsByStep = useMemo(() => getRowsByStep(run), [run]);
  const field = useMemo(() => computeRidgeGradientField(run, pca, step), [run, pca, step]);
  const details = useMemo(() => ridgeStepDetails(run, step), [run, step]);
  const addedRow = rowsByStep.get(step);
  const coverage = pca.explained.reduce((sum, value) => sum + value, 0);

  if (!run || !pca.projections.length) {
    return <main className="stage empty">No ridge snapshots available</main>;
  }

  return (
    <main className="stage ridgeStage">
      <div className="vizHeader">
        <div>
          <h1>Ridge Gradient Field</h1>
          <p className="sectionMeta">Ridge path in PCA space with gradient field for the selected t</p>
        </div>
        <div className="metricStrip ridgeMetrics">
          <Metric label="Grad norm" value={short(field.currentGradientNorm, 6)} />
          <Metric label="Ridge loss" value={short(field.currentLoss, 6)} />
          <Metric label="Added lift" value={fmt(addedRow?.liftNll)} tone={Number(addedRow?.liftNll) >= 0 ? "good" : "bad"} />
          <Metric label="PC coverage" value={pct(coverage)} />
        </div>
      </div>

      <div className={`ridgeLayout ${readoutOpen ? "readoutOpen" : "readoutCollapsed"}`}>
        <div className="ridgeSceneWrap">
          <div className="gradientLegend">
            <span>low gradient</span>
            <i>
              <em />
              <em />
              <em />
            </i>
            <span>high gradient</span>
          </div>
          <div className="axisLegend">
            {AXIS_LABELS.map(({ key }, index) => (
              <span key={key}>{axisLabelText(key, pca.featureLoadings?.[index])}</span>
            ))}
          </div>
          <RidgeScene pca={pca} field={field} rowsByStep={rowsByStep} addedRow={addedRow} setHover={setHover} />
          {hover && (
            <div className="ridgeTooltip" style={{ left: hover.x, top: hover.y }}>
              <strong>{hover.kind === "previous" ? "previous" : hover.kind === "current" ? "current" : "step"} t {hover.step}</strong>
              <span>{hover.candidateId || "unknown candidate"}</span>
              <b className={Number(hover.lift) >= 0 ? "green" : "red"}>{fmt(hover.lift)}</b>
              {hover.kind === "current" && <span>grad {short(hover.gradientNorm, 6)}</span>}
              {hover.kind === "current" && <span>loss {short(hover.loss, 6)}</span>}
            </div>
          )}
        </div>

        <aside className={`ridgeReadout ${readoutOpen ? "open" : "collapsed"}`}>
          <button className="readoutToggle" type="button" onClick={() => setReadoutOpen((value) => !value)}>
            <span>{readoutOpen ? "Hide" : "Details"}</span>
          </button>
          <div className="readoutContent">
          <section>
            <h3>Current Weights</h3>
            <div className="changeList">
              {details.topWeights.map((item) => (
                <div className="changeRow" key={item.feature}>
                  <span>{item.feature}</span>
                  <b className={item.weight >= 0 ? "green" : "red"}>{fmt(item.weight, 5)}</b>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3>Gradient Residual</h3>
            <div className="changeList">
              {field.topGradientFeatures.map((item) => (
                <div className="changeRow" key={item.feature}>
                  <span>{item.feature}</span>
                  <b className={item.gradient >= 0 ? "green" : "red"}>{fmt(item.gradient, 6)}</b>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3>PC Loadings</h3>
            <div className="pcList">
              {pca.featureLoadings.map((features, index) => (
                <div className="pcBlock" key={`pc-${index + 1}`}>
                  <div>
                    <strong>PC{index + 1}</strong>
                    <span>{pct(pca.explained[index])}</span>
                  </div>
                  {features.map((item) => (
                    <p key={`${index}-${item.feature}`}>
                      <span>{item.feature}</span>
                      <b>{fmt(item.loading, 3)}</b>
                    </p>
                  ))}
                </div>
              ))}
            </div>
          </section>
          </div>
        </aside>
      </div>
    </main>
  );
}

function Metric({ label, value, tone }) {
  return (
    <div className={`metric ${tone || ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
