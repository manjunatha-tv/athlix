
import { useEffect, useRef, useState, useCallback } from "react";
import type { MutableRefObject } from "react";

import {
  AnimationAction,
  AnimationMixer,
  LoopRepeat,
  Object3D,
  Vector3,
} from "three";

import { clone as cloneSkeleton } from "three/examples/jsm/utils/SkeletonUtils.js";

import { Canvas, useFrame } from "@react-three/fiber";

import {
  ContactShadows,
  OrbitControls,
  useAnimations,
  useGLTF,
} from "@react-three/drei";

const MODEL_PATH = "/models/cricket-batter.glb";
const SAMPLE_RATE = 30;

const REFERENCE_BONES: Record<string, string> = {
  left_shoulder: "mixamorig:LeftArm_09",
  right_shoulder: "mixamorig:RightArm_020",
  left_elbow: "mixamorig:LeftForeArm_010",
  right_elbow: "mixamorig:RightForeArm_021",
  left_wrist: "mixamorig:LeftHand_00",
  right_wrist: "mixamorig:RightHand_022",
  left_hip: "mixamorig:LeftUpLeg_031",
  right_hip: "mixamorig:RightUpLeg_036",
  left_knee: "mixamorig:LeftLeg_032",
  right_knee: "mixamorig:RightLeg_037",
  left_ankle: "mixamorig:LeftFoot_033",
  right_ankle: "mixamorig:RightFoot_038",
};

type ExportHandler = (() => void) | null;

type ModelProps = {
  playerRef: MutableRefObject<AnimationAction | null>;
  batRef: MutableRefObject<AnimationAction | null>;
  playing: boolean;
  speed: number;
  onDuration: (duration: number) => void;
  onProgress: (progress: number) => void;
  onExportReady: (handler: ExportHandler) => void;
};

function normalizeName(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function downloadJSON(data: unknown, filename: string) {
  const blob = new Blob(
    [JSON.stringify(data, null, 2)],
    { type: "application/json" }
  );

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;

  document.body.appendChild(link);
  link.click();
  link.remove();

  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

function getRuntimeBones(root: Object3D) {
  const bones = new Map<string, Object3D>();
  const objects: string[] = [];
  const skeletons: string[][] = [];

  root.traverse((object: any) => {
    if (object.name) {
      objects.push(object.name);
    }

    if (object.isBone) {
      bones.set(object.name, object);
    }

    if (object.isSkinnedMesh && object.skeleton) {
      const skeletonBoneNames = object.skeleton.bones.map(
        (bone: Object3D) => bone.name
      );

      skeletons.push(skeletonBoneNames);

      for (const bone of object.skeleton.bones) {
        bones.set(bone.name, bone);
      }
    }
  });

  return {
    bones,
    objects,
    skeletons,
  };
}

function findBone(
  bones: Map<string, Object3D>,
  requestedName: string
) {
  const target = normalizeName(requestedName);

  // First try exact normalized matching.
  for (const [name, bone] of bones.entries()) {
    if (normalizeName(name) === target) {
      return bone;
    }
  }

  // Then try matching names that contain the requested name.
  for (const [name, bone] of bones.entries()) {
    const candidate = normalizeName(name);

    if (
      candidate.includes(target) ||
      target.includes(candidate)
    ) {
      if (candidate.length > 5) {
        return bone;
      }
    }
  }

  return undefined;
}

function BatterModel({
  playerRef,
  batRef,
  playing,
  speed,
  onDuration,
  onProgress,
  onExportReady,
}: ModelProps) {
  const { scene, animations } = useGLTF(MODEL_PATH);
  const { actions } = useAnimations(animations, scene);

  useEffect(() => {
    const player = actions["Virat Playing Cricket"];
    const bat = actions["Cricket Bat Animation"];

    if (!player) {
      console.error("Player animation not found.");
      return;
    }

    playerRef.current = player;
    batRef.current = bat ?? null;

    const duration = player.getClip().duration;
    onDuration(duration);

    player.reset();
    player.setLoop(LoopRepeat, Infinity);
    player.enabled = true;
    player.play();

    if (bat) {
      bat.reset();
      bat.setLoop(LoopRepeat, Infinity);
      bat.enabled = true;
      bat.play();
    }

    return () => {
      player.stop();
      bat?.stop();

      playerRef.current = null;
      batRef.current = null;
    };
  }, [actions, playerRef, batRef, onDuration]);

  useEffect(() => {
    if (playerRef.current) {
      playerRef.current.timeScale = speed;
      playerRef.current.paused = !playing;
    }

    if (batRef.current) {
      batRef.current.timeScale = speed;
      batRef.current.paused = !playing;
    }
  }, [playing, speed, playerRef, batRef]);

  useEffect(() => {
    const playerClip = animations.find(
      (clip) => clip.name === "Virat Playing Cricket"
    );

    const batClip = animations.find(
      (clip) => clip.name === "Cricket Bat Animation"
    );

    if (!playerClip) {
      console.error("Reference animation not found.");
      return;
    }

    const exportReference = () => {
      try {
        // Clone the model so the visible animation stays untouched.
        const referenceScene = cloneSkeleton(scene);

        // Inspect actual runtime bones.
        const inventory = getRuntimeBones(referenceScene);

        console.log("ATHIX RUNTIME BONES:");
        console.log([...inventory.bones.keys()]);
        console.log("ATHIX SKELETONS:", inventory.skeletons);

        // Resolve the requested landmarks.
        const resolvedBones: Record<string, Object3D> = {};
        const missingBones: string[] = [];

        for (const [landmark, requestedName] of Object.entries(
          REFERENCE_BONES
        )) {
          const bone = findBone(inventory.bones, requestedName);

          if (bone) {
            resolvedBones[landmark] = bone;

            console.log(
              `Matched ${landmark}: ${requestedName} -> ${bone.name}`
            );
          } else {
            missingBones.push(`${landmark}: ${requestedName}`);
          }
        }

        // If the mapping fails, export a diagnostic file.
        if (missingBones.length > 0) {
          const diagnostic = {
            project: "ATHIX",
            message: "Runtime bone mapping needs correction.",
            missing_bones: missingBones,
            available_bones: [...inventory.bones.keys()],
            available_objects: inventory.objects,
            skeletons: inventory.skeletons,
          };

          downloadJSON(
            diagnostic,
            "athix_model_bone_diagnostic.json"
          );

          alert(
            "Some bones could not be matched.\n\n" +
              "ATHIX has downloaded a diagnostic JSON file. " +
              "Send me that file so I can correct the mapping."
          );

          return;
        }

        // Create an independent animation mixer.
        const mixer = new AnimationMixer(referenceScene);

        mixer.clipAction(playerClip).play();

        if (batClip) {
          mixer.clipAction(batClip).play();
        }

        const duration = playerClip.duration;
        const totalFrames = Math.ceil(duration * SAMPLE_RATE) + 1;

        const frames = [];
        const position = new Vector3();

        // Sample the complete animation.
        for (let frame = 0; frame < totalFrames; frame++) {
          const timestamp = Math.min(
            frame / SAMPLE_RATE,
            duration
          );

          mixer.setTime(timestamp);
          referenceScene.updateMatrixWorld(true);

          const landmarks: Record<
            string,
            { x: number; y: number; z: number }
          > = {};

          for (const [name, bone] of Object.entries(resolvedBones)) {
            bone.getWorldPosition(position);

            landmarks[name] = {
              x: Number(position.x.toFixed(6)),
              y: Number(position.y.toFixed(6)),
              z: Number(position.z.toFixed(6)),
            };
          }

          frames.push({
            frame,
            timestamp_seconds: Number(timestamp.toFixed(4)),
            landmarks,
          });
        }

        mixer.stopAllAction();

        const referenceData = {
          project: "ATHIX",
          technique: "straight-drive",
          animation: playerClip.name,
          sample_rate: SAMPLE_RATE,
          duration_seconds: duration,
          total_frames: frames.length,
          coordinate_system: "3D model world coordinates",
          landmarks: Object.fromEntries(
            Object.entries(resolvedBones).map(([name, bone]) => [
              name,
              bone.name,
            ])
          ),
          frames,
        };

        downloadJSON(
          referenceData,
          "athix_straight_drive_reference.json"
        );

        console.log("ATHIX reference extraction successful.");
        console.log("Duration:", duration);
        console.log("Sample rate:", SAMPLE_RATE);
        console.log("Frames:", frames.length);
        console.log("Landmarks:", Object.keys(resolvedBones));
      } catch (error) {
        console.error("ATHIX export error:", error);

        alert(
          error instanceof Error
            ? error.message
            : "Reference extraction failed."
        );
      }
    };

    onExportReady(exportReference);

    return () => {
      onExportReady(null);
    };
  }, [scene, animations, onExportReady]);

  const progressTimer = useRef(0);

  useFrame((_, delta) => {
    const player = playerRef.current;

    if (!player || !playing) return;

    progressTimer.current += delta;

    if (progressTimer.current >= 0.08) {
      const duration = player.getClip().duration;

      onProgress((player.time % duration) / duration);

      progressTimer.current = 0;
    }
  });

  return (
    <primitive
      object={scene}
      position={[0, -1, 0]}
      scale={1.8}
    />
  );
}

export default function CricketScene() {
  const playerRef = useRef<AnimationAction | null>(null);
  const batRef = useRef<AnimationAction | null>(null);

  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState(0);

  const [exportHandler, setExportHandler] =
    useState<ExportHandler>(null);

  const handleExportReady = useCallback(
    (handler: ExportHandler) => {
      setExportHandler(() => handler);
    },
    []
  );

  const handleSeek = (value: number) => {
    if (!duration) return;

    const nextTime = (value / 100) * duration;

    if (playerRef.current) {
      playerRef.current.time = nextTime;
    }

    if (batRef.current) {
      batRef.current.time =
        (nextTime / duration) *
        batRef.current.getClip().duration;
    }

    setProgress(value / 100);
  };

  const handleRestart = () => {
    const player = playerRef.current;
    const bat = batRef.current;

    if (player) {
      player.reset();
      player.timeScale = speed;
      player.play();
      player.paused = false;
    }

    if (bat) {
      bat.reset();
      bat.timeScale = speed;
      bat.play();
      bat.paused = false;
    }

    setProgress(0);
    setPlaying(true);
  };

  const formatTime = (seconds: number) => {
    const safe = Number.isFinite(seconds) ? seconds : 0;
    return `0:${safe.toFixed(1).padStart(4, "0")}`;
  };

  return (
    <div className="athix-viewer">
      <div className="athix-canvas">
        <Canvas
          camera={{
            position: [3, 2.5, 5],
            fov: 40,
          }}
          dpr={[1, 2]}
        >
          <color attach="background" args={["#101112"]} />

          <ambientLight intensity={1.4} />

          <directionalLight
            position={[5, 8, 5]}
            intensity={2}
          />

          <BatterModel
            playerRef={playerRef}
            batRef={batRef}
            playing={playing}
            speed={speed}
            onDuration={setDuration}
            onProgress={setProgress}
            onExportReady={handleExportReady}
          />

          <ContactShadows
            position={[0, -1, 0]}
            opacity={0.35}
            scale={8}
            blur={2}
          />

          <OrbitControls
            makeDefault
            enableRotate
            enableZoom
            enablePan
            enableDamping
            dampingFactor={0.06}
            minDistance={2}
            maxDistance={12}
            minPolarAngle={0.1}
            maxPolarAngle={Math.PI - 0.1}
          />
        </Canvas>
      </div>

      <div className="athix-playback">
        <div className="playback-top">
          <div className="playback-main-actions">
            <button
              className="playback-primary"
              onClick={() => setPlaying((value) => !value)}
            >
              {playing ? "Pause" : "Play"}
            </button>

            <button
              className="playback-secondary"
              onClick={handleRestart}
            >
              Restart
            </button>

            <button
              className="playback-secondary"
              onClick={() => exportHandler?.()}
              disabled={!exportHandler}
            >
              Export Reference Data
            </button>
          </div>

          <div className="playback-speeds">
            {[0.25, 0.5, 1].map((value) => (
              <button
                key={value}
                className={
                  speed === value
                    ? "speed-button active"
                    : "speed-button"
                }
                onClick={() => setSpeed(value)}
              >
                {value}×
              </button>
            ))}
          </div>
        </div>

        <div className="playback-timeline">
          <span className="playback-time">
            {formatTime(progress * duration)}
          </span>

          <input
            className="playback-range"
            type="range"
            min="0"
            max="100"
            step="0.1"
            value={progress * 100}
            onChange={(event) =>
              handleSeek(Number(event.target.value))
            }
          />

          <span className="playback-time">
            {formatTime(duration)}
          </span>
        </div>

        <div className="playback-footer">
          <span>ATHIX · TECHNIQUE VIEWER</span>
          <span>{speed}× PLAYBACK SPEED</span>
        </div>
      </div>
    </div>
  );
}

useGLTF.preload(MODEL_PATH);
