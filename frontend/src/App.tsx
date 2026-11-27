
import { useEffect, useRef, useState } from "react";
import "./App.css";
import CricketScene from "./components/CricketScene";

// ==========================================
// TYPES
// ==========================================

type Technique = {
  id: string;
  name: string;
  category: string;
  difficulty: string;
  description: string;
  focus: string[];
};

type Page = "library" | "demo" | "upload";

type SimilarityFeature = {
  similarity_percent: number | null;
  mean_absolute_error_degrees: number | null;
  valid_samples: number;
  status: string;
};

type SimilarityResult = {
  status?: string;
  label?: string;
  overall_similarity_percent?: number | null;
  features_compared?: number;
  comparison_samples?: number;
  message?: string;
  feature_scores?: Record<string, SimilarityFeature>;
  limitations?: string[];
};

type UploadResponse = {
  status?: string;
  message?: string;
  filename?: string;
  technique?: string;
  size_mb?: number;
  analysis_status?: string;
  detail?: string;
  upload_id?: string;
  total_frames?: number;
  detected_frames?: number;
  video_validation?: string;
  landmarks_file?: string;
  joint_angles_file?: string;
  smoothed_angles_file?: string;
  movement_phases_file?: string;
  phase_frames_analysed?: number;
  phase_distribution?: Record<string, number>;
  average_joint_angles?: Record<string, number | null>;
  similarity?: SimilarityResult;
  pose_overlay?: {
    available: boolean;
    url: string | null;
    frame_count: number;
  };
  annotated_video?: {
    available: boolean;
    url: string | null;
    frames_rendered?: number;
    fps?: number | null;
  };
};

type PosePoint = {
  x: number;
  y: number;
  confidence: number;
};

type PoseFrame = {
  frame: number;
  timestamp_seconds: number;
  similarity_percent: number | null;
  keypoints: Record<string, PosePoint>;
};

type PoseOverlayData = {
  status: string;
  scoring_label: string;
  frames: PoseFrame[];
  skeleton_edges: string[][];
};

// ==========================================
// TECHNIQUES
// ==========================================

const techniques: Technique[] = [
  {
    id: "straight-drive",
    name: "Straight Drive",
    category: "DRIVING",
    difficulty: "Intermediate",
    description:
      "A front-foot attacking shot played straight down the ground with controlled balance and timing.",
    focus: ["Head position", "Front-foot movement", "Body balance"],
  },
  {
    id: "cover-drive",
    name: "Cover Drive",
    category: "DRIVING",
    difficulty: "Advanced",
    description:
      "An elegant front-foot shot directed through the cover region.",
    focus: ["Foot placement", "Head alignment", "Bat swing"],
  },
  {
    id: "pull-shot",
    name: "Pull Shot",
    category: "HORIZONTAL BAT",
    difficulty: "Advanced",
    description:
      "A horizontal-bat attacking shot used against short-pitched deliveries.",
    focus: ["Body rotation", "Weight transfer", "Bat control"],
  },
  {
    id: "cut-shot",
    name: "Cut Shot",
    category: "HORIZONTAL BAT",
    difficulty: "Intermediate",
    description:
      "A back-foot shot directed through the off side.",
    focus: ["Back-foot movement", "Balance", "Bat angle"],
  },
];

// ==========================================
// CONFIGURATION
// ==========================================

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const API_BASE_URL = "http://127.0.0.1:8000/api";
const API_URL = `${API_BASE_URL}/analyze`;

const ALLOWED_TYPES = [
  "video/mp4",
  "video/quicktime",
  "video/webm",
];

// ==========================================
// HELPERS
// ==========================================

function formatSize(bytes: number) {
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return "Unknown";

  const minutes = Math.floor(seconds / 60);
  const remaining = Math.floor(seconds % 60);

  return `${minutes}:${String(remaining).padStart(2, "0")}`;
}

function formatLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

// ==========================================
// MAIN APP
// ==========================================

function App() {
  const [selected, setSelected] = useState<Technique>(techniques[0]);
  const [page, setPage] = useState<Page>("library");

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [videoDuration, setVideoDuration] = useState(0);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] =
    useState<UploadResponse | null>(null);

  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [battingHand, setBattingHand] = useState<"right" | "left">("right");
  const [cameraView, setCameraView] = useState<"side" | "front" | "three-quarter">("side");
  const [poseOverlay, setPoseOverlay] = useState<PoseOverlayData | null>(null);
  const [currentFrameScore, setCurrentFrameScore] = useState<number | null>(null);
  const [overlayLoading, setOverlayLoading] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(0.5);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const originalVideoRef = useRef<HTMLVideoElement>(null);

  // ========================================
  // VIDEO PREVIEW URL
  // ========================================

  useEffect(() => {
    if (!videoFile) {
      setVideoUrl("");
      return;
    }

    const url = URL.createObjectURL(videoFile);
    setVideoUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [videoFile]);

  // ========================================
  // LOAD FRAME-BY-FRAME YOLO OVERLAY DATA
  // ========================================

  useEffect(() => {
    const overlayUrl = uploadResult?.pose_overlay?.url;

    if (!overlayUrl || !uploadResult.pose_overlay?.available) {
      setPoseOverlay(null);
      setCurrentFrameScore(null);
      return;
    }

    let cancelled = false;
    setOverlayLoading(true);

    fetch(`http://127.0.0.1:8000${overlayUrl}`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Could not load pose overlay (${response.status}).`);
        }
        return (await response.json()) as PoseOverlayData;
      })
      .then((data) => {
        if (!cancelled) {
          setPoseOverlay(data);
        }
      })
      .catch((overlayError) => {
        console.error("ATHIX pose overlay error:", overlayError);
        if (!cancelled) {
          setPoseOverlay(null);
        }
      })
      .finally(() => {
        if (!cancelled) setOverlayLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [uploadResult?.pose_overlay?.url, uploadResult?.pose_overlay?.available]);

  // ========================================
  // SYNC ORIGINAL VIDEO WITH ANALYSED VIDEO
  // ========================================

  useEffect(() => {
    const analysed = videoRef.current;
    const original = originalVideoRef.current;
    if (!analysed || !original) return;

    const syncOriginal = () => {
      if (Math.abs(original.currentTime - analysed.currentTime) > 0.08) {
        original.currentTime = analysed.currentTime;
      }
      original.playbackRate = analysed.playbackRate;
      setCurrentVideoTime(analysed.currentTime);
    };
    const playOriginal = () => {
      syncOriginal();
      setIsVideoPlaying(true);
      void original.play().catch(() => undefined);
    };
    const pauseOriginal = () => {
      setIsVideoPlaying(false);
      original.pause();
    };
    const seekOriginal = () => syncOriginal();
    const rateOriginal = () => {
      original.playbackRate = analysed.playbackRate;
      setPlaybackSpeed(analysed.playbackRate);
    };

    analysed.playbackRate = playbackSpeed;
    original.playbackRate = playbackSpeed;
    analysed.addEventListener("play", playOriginal);
    analysed.addEventListener("pause", pauseOriginal);
    analysed.addEventListener("seeking", seekOriginal);
    analysed.addEventListener("seeked", seekOriginal);
    analysed.addEventListener("timeupdate", syncOriginal);
    analysed.addEventListener("ratechange", rateOriginal);

    return () => {
      analysed.removeEventListener("play", playOriginal);
      analysed.removeEventListener("pause", pauseOriginal);
      analysed.removeEventListener("seeking", seekOriginal);
      analysed.removeEventListener("seeked", seekOriginal);
      analysed.removeEventListener("timeupdate", syncOriginal);
      analysed.removeEventListener("ratechange", rateOriginal);
    };
  }, [videoUrl, uploadResult, playbackSpeed]);

  // ========================================
  // SYNC FRAME SCORE TO ANALYSED VIDEO
  // ========================================

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !poseOverlay?.frames?.length) return;

    const updateScore = () => {
      const frames = poseOverlay.frames;
      const time = video.currentTime;
      let low = 0;
      let high = frames.length - 1;
      while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (frames[middle].timestamp_seconds < time) low = middle + 1;
        else high = middle;
      }
      const next = frames[low];
      const previous = frames[Math.max(0, low - 1)];
      const nearest = Math.abs(previous.timestamp_seconds - time) <= Math.abs(next.timestamp_seconds - time)
        ? previous : next;
      setCurrentFrameScore(nearest.similarity_percent);
    };

    video.addEventListener("timeupdate", updateScore);
    video.addEventListener("seeked", updateScore);
    video.addEventListener("loadedmetadata", updateScore);
    updateScore();
    return () => {
      video.removeEventListener("timeupdate", updateScore);
      video.removeEventListener("seeked", updateScore);
      video.removeEventListener("loadedmetadata", updateScore);
    };
  }, [poseOverlay, uploadResult?.annotated_video?.url]);

  function toggleVideoPlayback() {
    const analysed = videoRef.current;
    if (!analysed) return;
    if (analysed.paused) {
      void analysed.play().catch((playError) => {
        console.error("ATHIX video playback error:", playError);
      });
    } else {
      analysed.pause();
    }
  }

  function seekVideo(time: number) {
    const analysed = videoRef.current;
    const original = originalVideoRef.current;
    if (!analysed) return;
    const safeTime = Math.max(0, Math.min(time, Number.isFinite(analysed.duration) ? analysed.duration : time));
    analysed.currentTime = safeTime;
    if (original) original.currentTime = safeTime;
    setCurrentVideoTime(safeTime);
  }

  function changePlaybackSpeed(speed: number) {
    setPlaybackSpeed(speed);
    if (videoRef.current) videoRef.current.playbackRate = speed;
    if (originalVideoRef.current) originalVideoRef.current.playbackRate = speed;
  }

  // ========================================
  // NAVIGATION
  // ========================================

  function openDemo() {
    setPage("demo");
  }

  function goToLibrary() {
    if (isUploading) return;
    setPage("library");
  }

  function openUpload() {
    setError("");
    setUploadResult(null);
    setPage("upload");
  }

  // ========================================
  // VIDEO VALIDATION
  // ========================================

  function validateFile(file: File) {
    const extensionValid = /\.(mp4|mov|webm)$/i.test(file.name);

    const typeValid =
      ALLOWED_TYPES.includes(file.type) || extensionValid;

    if (!typeValid) {
      setError("Please select an MP4, MOV or WebM video.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setError("Your video exceeds the 20 MB file limit.");
      return;
    }

    if (file.size === 0) {
      setError("This video file is empty.");
      return;
    }

    setError("");
    setUploadResult(null);
    setPoseOverlay(null);
    setCurrentFrameScore(null);
    setVideoDuration(0);
    setVideoFile(file);
  }

  // ========================================
  // FILE INPUT
  // ========================================

  function handleFileChange(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];

    if (file) {
      validateFile(file);
    }

    event.target.value = "";
  }

  // ========================================
  // DRAG AND DROP
  // ========================================

  function handleDrop(
    event: React.DragEvent<HTMLDivElement>
  ) {
    event.preventDefault();
    setDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      validateFile(file);
    }
  }

  // ========================================
  // REMOVE VIDEO
  // ========================================

  function removeVideo() {
    if (isUploading) return;

    setVideoFile(null);
    setVideoDuration(0);
    setError("");
    setUploadResult(null);
    setPoseOverlay(null);
    setCurrentFrameScore(null);
  }

  // ========================================
  // BACKEND ANALYSIS
  // ========================================

  async function handleAnalysis() {
    if (!videoFile || isUploading) return;

    setIsUploading(true);
    setError("");
    setUploadResult(null);
    setPoseOverlay(null);
    setCurrentFrameScore(null);

    try {
      const formData = new FormData();

      formData.append("video", videoFile);
      formData.append("technique", selected.id);
      formData.append("batting_hand", battingHand);
      formData.append("camera_view", cameraView);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data: UploadResponse = await response
        .json()
        .catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          data.detail ||
            `Analysis failed. Server returned ${response.status}.`
        );
      }

      setUploadResult(data);

      console.log("ATHIX ANALYSIS RESPONSE:", data);
    } catch (uploadError) {
      console.error("ATHIX analysis error:", uploadError);

      setError(
        uploadError instanceof Error
          ? uploadError.message
          : "Could not connect to the ATHIX backend."
      );
    } finally {
      setIsUploading(false);
    }
  }

  // RENDER
  // ========================================

  const annotatedVideoUrl = uploadResult?.annotated_video?.available && uploadResult.annotated_video.url
    ? `http://127.0.0.1:8000${uploadResult.annotated_video.url}`
    : "";

  return (
    <main className="app">
      <style>{`@media (max-width: 900px) { .athix-video-comparison { grid-template-columns: 1fr !important; } .upload-page { width: calc(100vw - 28px) !important; } }`}</style>

      {/* NAVIGATION */}

      <nav className="navbar">
        <div className="brand">
          <span className="brand-mark">A</span>
          <span>ATHIX</span>
        </div>

        <div className="nav-links">
          <button
            className={page === "library" ? "nav-active" : ""}
            onClick={goToLibrary}
          >
            Technique Library
          </button>

          <button onClick={goToLibrary}>
            My Analysis
          </button>
        </div>

        <div className="nav-right">
          <span className="status-dot" />
          <span>CRICKET</span>
        </div>
      </nav>

      {/* TECHNIQUE LIBRARY */}

      {page === "library" && (
        <section className="content">

          <div className="eyebrow">
            ATHIX / CRICKET / BATTING
          </div>

          <div className="hero">
            <div>
              <p className="overline">
                TRAIN WITH INTELLIGENCE
              </p>

              <h1>
                Master your
                <br />
                batting technique.
              </h1>

              <p className="hero-description">
                Learn the fundamentals. Study professional movement.
                Analyse your technique and improve with every attempt.
              </p>
            </div>

            <div className="hero-visual">
              <div className="visual-circle">
                <span>01</span>
                <div className="visual-line" />

                <strong>
                  BAT
                  <br />
                  BETTER.
                </strong>

                <small>PERFORMANCE ANALYSIS</small>
              </div>
            </div>
          </div>

          <div className="section-heading">
            <div>
              <p className="overline">
                YOUR TRAINING STARTS HERE
              </p>

              <h2>Batting techniques</h2>
            </div>

            <span className="count">04 TECHNIQUES</span>
          </div>

          <div className="technique-layout">

            <div className="technique-grid">
              {techniques.map((technique, index) => (
                <button
                  key={technique.id}
                  className={`technique-card ${
                    selected.id === technique.id ? "selected" : ""
                  }`}
                  onClick={() => setSelected(technique)}
                >
                  <div className="card-top">
                    <span>0{index + 1}</span>

                    <span className="select-indicator">
                      {selected.id === technique.id
                        ? "SELECTED"
                        : "+"}
                    </span>
                  </div>

                  <div className="shot-symbol">
                    {index < 2
                      ? "↗"
                      : index === 2
                        ? "↙"
                        : "↖"}
                  </div>

                  <div>
                    <span className="category">
                      {technique.category}
                    </span>

                    <h3>{technique.name}</h3>
                    <p>{technique.difficulty}</p>
                  </div>
                </button>
              ))}
            </div>

            <aside className="details-panel">
              <span className="overline">
                SELECTED TECHNIQUE
              </span>

              <h2>{selected.name}</h2>

              <p className="detail-description">
                {selected.description}
              </p>

              <div className="divider" />

              <span className="overline">
                KEY AREAS TO STUDY
              </span>

              <div className="focus-list">
                {selected.focus.map((item, index) => (
                  <div className="focus-item" key={item}>
                    <span>0{index + 1}</span>
                    {item}
                  </div>
                ))}
              </div>

              <button
                className="primary-button"
                onClick={openDemo}
              >
                Study technique <span>→</span>
              </button>
            </aside>

          </div>
        </section>
      )}

      {/* TECHNIQUE DEMONSTRATION */}

      {page === "demo" && (
        <section className="content demo-page">

          <button
            className="back-button"
            onClick={goToLibrary}
          >
            ← Back to techniques
          </button>

          <p className="overline">
            TECHNIQUE GUIDE / CRICKET
          </p>

          <h1>{selected.name}</h1>

          <p className="hero-description">
            Study the movement before recording your own attempt.
          </p>

          {selected.id === "straight-drive" ? (
            <>
              <div className="animation-layout">

                <div className="animation-stage">
                  <div className="animation-label">
                    <span>3D TECHNIQUE VIEWER</span>
                    <span>STRAIGHT DRIVE</span>
                  </div>

                  <CricketScene />
                </div>

                <aside className="phase-panel">
                  <span className="overline">
                    3D DEMONSTRATION
                  </span>

                  <h2>Cricket batter</h2>

                  <p>
                    Explore the 3D player using your mouse.
                    Drag to rotate the camera and scroll to zoom.
                  </p>

                  <div className="phase-tip">
                    <span>REFERENCE MODEL</span>
                    Explore the player's batting movement
                    from different viewing angles.
                  </div>
                </aside>

              </div>

              <div className="learning-note">
                <span className="overline">
                  DEVELOPMENT PREVIEW
                </span>

                <p>
                  The reference animation is available.
                  Video-based technique analysis is under development.
                </p>
              </div>
            </>
          ) : (
            <div className="demo-placeholder">
              <span>DEMONSTRATION IN DEVELOPMENT</span>
              <div className="play-symbol">↗</div>

              <p>
                We are developing this technique's 3D demonstration next.
              </p>
            </div>
          )}

          <button
            className="primary-button upload-button"
            onClick={openUpload}
          >
            Continue to video upload →
          </button>

        </section>
      )}

      {/* VIDEO UPLOAD PAGE */}

      {page === "upload" && (
        <section className="content upload-page" style={{ width: "min(1600px, calc(100vw - 64px))", maxWidth: "1600px", marginInline: "auto" }}>

          <button
            className="back-button"
            onClick={() => {
              if (!isUploading) setPage("demo");
            }}
          >
            ← Back to demonstration
          </button>

          <div className="upload-heading">
            <p className="overline">
              ATHIX / VIDEO ANALYSIS
            </p>

            <h1>
              Upload your
              <br />
              <span>batting attempt.</span>
            </h1>

            <p className="hero-description">
              Your technique journey starts here.
              Upload a recording to analyse your movement.
            </p>
          </div>

          {/* SELECTED TECHNIQUE */}

          <div className="upload-context">
            <div>
              <span className="overline">
                SELECTED TECHNIQUE
              </span>

              <h2>{selected.name}</h2>
              <p>CRICKET / BATTING</p>
            </div>

            <div className="upload-context-tag">
              VIDEO ANALYSIS
            </div>
          </div>

          {/* CAMERA AND BATTING-HAND SETTINGS */}

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "14px", margin: "20px 0", padding: "18px", border: "1px solid #343434", borderRadius: "8px", background: "#111111" }}>
            <label style={{ display: "grid", gap: "8px", color: "#aaaaaa", fontSize: "12px" }}>
              BATTING HAND
              <select value={battingHand} onChange={(event) => setBattingHand(event.target.value as "right" | "left")} style={{ padding: "12px", color: "#eeeeee", background: "#1b1b1b", border: "1px solid #444", borderRadius: "6px" }}>
                <option value="right">Right-handed</option>
                <option value="left">Left-handed</option>
              </select>
            </label>
            <label style={{ display: "grid", gap: "8px", color: "#aaaaaa", fontSize: "12px" }}>
              CAMERA VIEW
              <select value={cameraView} onChange={(event) => setCameraView(event.target.value as "side" | "front" | "three-quarter")} style={{ padding: "12px", color: "#eeeeee", background: "#1b1b1b", border: "1px solid #444", borderRadius: "6px" }}>
                <option value="side">Side view</option>
                <option value="front">Front view</option>
                <option value="three-quarter">Three-quarter view</option>
              </select>
            </label>
          </div>

          {/* FILE INPUT */}

          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.mov,.webm,video/mp4,video/quicktime,video/webm"
            onChange={handleFileChange}
            hidden
          />

          {/* UPLOAD DROPZONE */}

          {!videoFile ? (
            <div
              className={`upload-dropzone ${
                dragging ? "dragging" : ""
              }`}
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <div className="upload-icon">
                <svg
                  width="27"
                  height="27"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 16V4" />
                  <path d="m7 9 5-5 5 5" />
                  <path d="M4 16v4h16v-4" />
                </svg>
              </div>

              <h2>Drop your video here</h2>

              <p>
                Drag and drop your recording or browse files
                from your device.
              </p>

              <span className="browse-button">
                Browse videos
              </span>

              <small>
                MP4, MOV or WebM · Maximum file size 20 MB
              </small>
            </div>
          ) : (
            <div className="video-preview-card">

              <div className="video-preview-header">
                <div>
                  <span className="overline">
                    VIDEO SELECTED
                  </span>

                  <h3>Your recording</h3>
                </div>

                <button
                  className="remove-video-button"
                  onClick={removeVideo}
                  disabled={isUploading}
                >
                  Remove video
                </button>
              </div>

              <div className="video-file-details">
                <div>
                  <span>FILE NAME</span>
                  <strong>{videoFile.name}</strong>
                </div>

                <div>
                  <span>FILE SIZE</span>
                  <strong>{formatSize(videoFile.size)}</strong>
                </div>

                <div>
                  <span>DURATION</span>
                  <strong>{formatDuration(videoDuration)}</strong>
                </div>
              </div>

              <button
                className="replace-video-button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
              >
                Replace video
              </button>
            </div>
          )}

          {/* ERROR */}

          {error && (
            <div className="upload-error" role="alert">
              <span>!</span>
              {error}
            </div>
          )}

          {/* SIDE-BY-SIDE VIDEO REVIEW — ABOVE THE RESULTS */}
          {videoFile && (
            <section
              className="video-review-section"
              style={{ marginTop: "28px", padding: "22px", border: "1px solid #292d25", borderRadius: "16px", background: "linear-gradient(180deg, #121410 0%, #0b0c0a 100%)" }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "end", gap: "12px", flexWrap: "wrap", marginBottom: "4px" }}>
                <div>
                  <span className="overline">VIDEO REVIEW / FRAME-BY-FRAME</span>
                  <h2 style={{ color: "#f0f2eb", fontSize: "22px", margin: "8px 0 0" }}>Compare your movement</h2>
                </div>
                <span style={{ color: "#8d9581", fontSize: "11px" }}>Original vs YOLO analysis</span>
              </div>
                            <div
                className="athix-video-comparison"
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                  gap: "22px",
                  marginTop: "14px",
                  width: "100%",
                }}
              >
                {/* ORIGINAL VIDEO — LEFT */}
                <div style={{ minWidth: 0, padding: "14px", border: "1px solid #292d25", borderRadius: "12px", background: "linear-gradient(180deg, #151713 0%, #0d0e0c 100%)", boxShadow: "0 14px 36px rgba(0,0,0,0.22)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
                    <span className="overline">01 / ORIGINAL VIDEO</span>
                    <span style={{ color: "#888", fontSize: "11px" }}>Unmodified recording</span>
                  </div>
                  <div style={{ overflow: "hidden", borderRadius: "8px", background: "#050505", aspectRatio: "16 / 9" }}>
                    <video
                      key={`original-${videoUrl}`}
                      ref={originalVideoRef}
                      src={videoUrl}
                      playsInline
                      muted
                      preload="metadata"
                      style={{ display: "block", width: "100%", height: "100%", objectFit: "contain" }}
                      onLoadedMetadata={(event) => {
                        setVideoDuration(event.currentTarget.duration);
                        event.currentTarget.playbackRate = playbackSpeed;
                      }}
                    />
                  </div>
                </div>

                {/* ANALYSED VIDEO — RIGHT */}
                <div style={{ minWidth: 0, padding: "14px", border: "1px solid rgba(217,255,99,0.28)", borderRadius: "12px", background: "linear-gradient(180deg, #171c12 0%, #0d0e0c 100%)", boxShadow: "0 14px 36px rgba(0,0,0,0.22)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px", marginBottom: "12px", flexWrap: "wrap" }}>
                    <span className="overline">02 / YOLO POSE ANALYSIS</span>
                  </div>
                  <div style={{ position: "relative", width: "100%", overflow: "hidden", borderRadius: "8px", background: "#050505", aspectRatio: "16 / 9" }}>
                    {annotatedVideoUrl ? (
                      <video
                        key={`analysed-${annotatedVideoUrl}`}
                        ref={videoRef}
                        src={annotatedVideoUrl}
                        playsInline
                        preload="metadata"
                        style={{ display: "block", width: "100%", height: "100%", objectFit: "contain" }}
                        onLoadedMetadata={(event) => {
                          event.currentTarget.playbackRate = playbackSpeed;
                          setVideoDuration(event.currentTarget.duration || 0);
                          setCurrentVideoTime(event.currentTarget.currentTime || 0);
                        }}
                      />
                    ) : (
                      <div style={{ width: "100%", height: "100%", minHeight: "220px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "12px", padding: "24px", textAlign: "center", background: "radial-gradient(circle at 50% 35%, #202719 0%, #090b08 72%)" }}>
                        <span style={{ color: "#d9ff63", fontSize: "30px" }}>◎</span>
                        <strong style={{ color: "#f0f2eb", fontSize: "14px" }}>YOLO annotated video</strong>
                        <span style={{ maxWidth: "280px", color: "#92998a", fontSize: "11px", lineHeight: 1.6 }}>{uploadResult ? "The backend did not return an annotated video. Restart the updated backend and analyse this clip again." : "Your detected skeleton and frame-by-frame score will appear here after analysis."}</span>
                      </div>
                    )}
                    {annotatedVideoUrl && uploadResult && (
                      <div style={{ position: "absolute", top: "12px", left: "12px", zIndex: 2, padding: "9px 12px", border: "1px solid rgba(217,255,99,0.55)", borderRadius: "8px", background: "rgba(8,12,6,0.88)", backdropFilter: "blur(8px)", color: "#d9ff63", minWidth: "108px", pointerEvents: "none" }}>
                        <span style={{ display: "block", fontSize: "9px", letterSpacing: "1.3px", color: "#c0c9a8" }}>FRAME SIMILARITY</span>
                        <strong style={{ display: "block", fontSize: "28px", lineHeight: 1.2, marginTop: "3px" }}>{currentFrameScore == null ? "—" : `${currentFrameScore.toFixed(1)}%`}</strong>
                        <span style={{ display: "block", fontSize: "9px", color: "#aab394", marginTop: "3px" }}>EXPERIMENTAL</span>
                      </div>
                    )}
                    {overlayLoading && (
                      <div style={{ position: "absolute", right: "12px", top: "12px", padding: "7px 10px", borderRadius: "6px", background: "rgba(0,0,0,0.75)", color: "#ddd", fontSize: "11px" }}>Loading frame scores…</div>
                    )}
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", gap: "10px", alignItems: "center", marginTop: "12px", flexWrap: "wrap" }}>
                    <span style={{ color: "#a9b396", fontSize: "10px", letterSpacing: "1px" }}>YOLO KEYPOINTS + JOINT CONNECTIONS</span>
                    <span style={{ color: annotatedVideoUrl ? "#d9ff63" : "#d6a45a", fontSize: "10px" }}>{annotatedVideoUrl ? "ANNOTATED VIDEO READY" : "ANNOTATED VIDEO NOT AVAILABLE"}</span>
                  </div>
                  <p style={{ color: "#888", fontSize: "11px", lineHeight: 1.5, marginTop: "7px" }}>The original recording stays synchronised with these shared controls.</p>
                </div>
              </div>

              {annotatedVideoUrl && (
                <div className="athix-shared-video-controls" style={{ width: "min(760px, 100%)", margin: "18px auto 4px", padding: "16px 20px", border: "1px solid rgba(217,255,99,0.24)", borderRadius: "14px", background: "linear-gradient(135deg, #171b12 0%, #10110e 100%)", boxShadow: "0 16px 40px rgba(0,0,0,0.28)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px", marginBottom: "12px" }}>
                    <span style={{ color: "#d9ff63", fontSize: "10px", letterSpacing: "1.5px", fontWeight: 700 }}>SHARED VIDEO CONTROLS</span>
                    <span style={{ color: "#a8ad9b", fontSize: "11px", fontVariantNumeric: "tabular-nums" }}>{formatDuration(currentVideoTime)} / {formatDuration(videoDuration)}</span>
                  </div>
                  <input aria-label="Seek both videos" type="range" min={0} max={Math.max(videoDuration, 0.01)} step={0.01} value={Math.min(currentVideoTime, videoDuration || 0)} onChange={(event) => seekVideo(Number(event.target.value))} style={{ width: "100%", accentColor: "#d9ff63", cursor: "pointer", margin: "2px 0 14px" }} />
                  <div style={{ display: "flex", justifyContent: "center", alignItems: "center", gap: "18px", flexWrap: "wrap" }}>
                    <button type="button" onClick={() => seekVideo(currentVideoTime - 1)} aria-label="Go back one second" style={{ border: "1px solid #3a3e32", background: "#1a1d16", color: "#e6e9df", borderRadius: "9px", padding: "10px 13px", cursor: "pointer" }}>−1s</button>
                    <button type="button" onClick={toggleVideoPlayback} aria-label={isVideoPlaying ? "Pause both videos" : "Play both videos"} style={{ width: "48px", height: "48px", border: 0, borderRadius: "50%", background: "#d9ff63", color: "#111", fontSize: "18px", fontWeight: 800, cursor: "pointer", boxShadow: "0 0 24px rgba(217,255,99,0.18)" }}>{isVideoPlaying ? "Ⅱ" : "▶"}</button>
                    <button type="button" onClick={() => seekVideo(currentVideoTime + 1)} aria-label="Forward one second" style={{ border: "1px solid #3a3e32", background: "#1a1d16", color: "#e6e9df", borderRadius: "9px", padding: "10px 13px", cursor: "pointer" }}>+1s</button>
                    <label style={{ display: "flex", alignItems: "center", gap: "9px", color: "#a8ad9b", fontSize: "11px" }}>SPEED <select value={playbackSpeed} onChange={(event) => changePlaybackSpeed(Number(event.target.value))} style={{ padding: "10px 12px", color: "#d9ff63", background: "#171717", border: "1px solid #454545", borderRadius: "8px", cursor: "pointer" }}>
                      <option value={0.25}>0.25×</option><option value={0.5}>0.50×</option><option value={0.75}>0.75×</option><option value={1}>1.00×</option><option value={1.25}>1.25×</option><option value={1.5}>1.50×</option>
                    </select></label>
                  </div>
                  <p style={{ color: "#777d6d", fontSize: "10px", textAlign: "center", margin: "12px 0 0" }}>One control panel operates both recordings together.</p>
                </div>
              )}

              {uploadResult && !annotatedVideoUrl && (
                <p style={{ color: "#d6a45a", fontSize: "12px", marginTop: "12px", padding: "12px 14px", border: "1px solid #59472b", borderRadius: "8px", background: "#17130d" }}>
                  {overlayLoading ? "Loading YOLO analysis data…" : "The backend did not generate the annotated video. Replace main.py with the latest backend, restart it, and analyse this clip again."}
                </p>
              )}


            </section>
          )}


          {/* ANALYSIS RESULTS */}

          {uploadResult && (
            <div className="upload-success" role="status">

              <div className="upload-success-heading">
                <span>✓</span>
                <strong>Movement analysis completed</strong>
              </div>

              <p>
                {uploadResult.message ||
                  "ATHIX successfully processed your video."}
              </p>

              {/* PROCESSING SUMMARY */}

              <div className="upload-result-details">
                <div>
                  <span>FRAMES PROCESSED</span>
                  <strong>{uploadResult.total_frames ?? "—"}</strong>
                </div>

                <div>
                  <span>PERSON DETECTIONS</span>
                  <strong>{uploadResult.detected_frames ?? "—"}</strong>
                </div>

                <div>
                  <span>VIDEO VALIDATION</span>
                  <strong>
                    {uploadResult.video_validation === "passed"
                      ? "Passed"
                      : uploadResult.video_validation || "—"}
                  </strong>
                </div>
              </div>

              {/* AVERAGE JOINT ANGLES */}

              {uploadResult.average_joint_angles && (
                <div style={{ marginTop: "30px" }}>
                  <span className="overline">
                    AVERAGE JOINT ANGLES
                  </span>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns:
                        "repeat(auto-fit, minmax(180px, 1fr))",
                      gap: "12px",
                      marginTop: "16px",
                    }}
                  >
                    {Object.entries(
                      uploadResult.average_joint_angles
                    ).map(([joint, angle]) => (
                      <div
                        key={joint}
                        style={{
                          padding: "14px",
                          borderBottom: "1px solid #333333",
                        }}
                      >
                        <span
                          style={{
                            display: "block",
                            fontSize: "11px",
                            color: "#999999",
                            textTransform: "uppercase",
                            marginBottom: "7px",
                          }}
                        >
                          {formatLabel(joint)}
                        </span>

                        <strong style={{ fontSize: "21px" }}>
                          {angle === null
                            ? "Unavailable"
                            : `${angle}°`}
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* REFERENCE SIMILARITY */}

              <div style={{ marginTop: "34px" }}>
                <span className="overline">
                  REFERENCE TECHNIQUE COMPARISON
                </span>

                {uploadResult.similarity?.overall_similarity_percent != null ? (
                  <>
                    <div
                      style={{
                        marginTop: "16px",
                        padding: "24px",
                        border: "1px solid #52652b",
                        borderRadius: "10px",
                        background: "#171d12",
                      }}
                    >
                      <span
                        style={{
                          color: "#b8c0a7",
                          fontSize: "11px",
                          letterSpacing: "1.5px",
                          textTransform: "uppercase",
                        }}
                      >
                        {uploadResult.similarity.label || "Experimental similarity"}
                      </span>

                      <div
                        style={{
                          display: "flex",
                          alignItems: "baseline",
                          gap: "8px",
                          marginTop: "10px",
                        }}
                      >
                        <strong style={{ fontSize: "48px", color: "#d9ff63" }}>
                          {uploadResult.similarity.overall_similarity_percent.toFixed(1)}
                        </strong>
                        <span style={{ color: "#d9ff63", fontSize: "20px" }}>%</span>
                      </div>

                      <p style={{ color: "#b8c0a7", marginTop: "8px" }}>
                        Compared across {uploadResult.similarity.features_compared ?? 0} joint-angle patterns
                        using {uploadResult.similarity.comparison_samples ?? 100} normalized time samples.
                      </p>
                    </div>

                    {uploadResult.similarity.feature_scores && (
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                          gap: "12px",
                          marginTop: "16px",
                        }}
                      >
                        {Object.entries(uploadResult.similarity.feature_scores).map(
                          ([joint, result]: [string, SimilarityFeature]) => (
                            <div
                              key={joint}
                              style={{
                                padding: "15px",
                                border: "1px solid #343434",
                                borderRadius: "8px",
                                background: "#111111",
                              }}
                            >
                              <span
                                style={{
                                  display: "block",
                                  color: "#999999",
                                  fontSize: "11px",
                                  textTransform: "uppercase",
                                  marginBottom: "9px",
                                }}
                              >
                                {formatLabel(joint)}
                              </span>

                              <strong style={{ fontSize: "23px", color: "#d9ff63" }}>
                                {result.similarity_percent == null
                                  ? "Unavailable"
                                  : `${result.similarity_percent.toFixed(1)}%`}
                              </strong>

                              <p style={{ color: "#999999", fontSize: "12px", marginTop: "7px" }}>
                                {result.mean_absolute_error_degrees == null
                                  ? "Insufficient valid samples"
                                  : `Mean angle error: ${result.mean_absolute_error_degrees.toFixed(1)}°`}
                              </p>
                            </div>
                          )
                        )}
                      </div>
                    )}

                    <div
                      style={{
                        marginTop: "16px",
                        padding: "16px",
                        borderLeft: "3px solid #d9ff63",
                        background: "#171a12",
                      }}
                    >
                      <strong style={{ color: "#d9ff63", fontSize: "12px" }}>
                        EXPERIMENTAL RESULT — NOT A COACHING GRADE
                      </strong>
                      <p style={{ color: "#b8b8b8", marginTop: "8px", lineHeight: 1.7 }}>
                        This percentage measures similarity to the saved trainer video using
                        estimated 2D joint-angle patterns. It is not proof that the technique is correct.
                        Camera perspective, pose-estimation errors and recording differences can affect the result.
                      </p>
                      {uploadResult.similarity.limitations?.length ? (
                        <ul style={{ color: "#999999", fontSize: "12px", lineHeight: 1.7, paddingLeft: "18px", marginTop: "10px" }}>
                          {uploadResult.similarity.limitations.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  </>
                ) : (
                  <div
                    style={{
                      marginTop: "16px",
                      padding: "18px",
                      border: "1px solid #343434",
                      borderRadius: "8px",
                      background: "#111111",
                    }}
                  >
                    <strong style={{ color: "#d9ff63" }}>
                      Similarity score unavailable
                    </strong>
                    <p style={{ color: "#aaaaaa", marginTop: "8px", lineHeight: 1.6 }}>
                      {uploadResult.similarity?.message ||
                        "The backend did not return a usable reference comparison for this technique."}
                    </p>
                    <p style={{ color: "#888888", fontSize: "12px", marginTop: "8px" }}>
                      A saved trainer reference is required for this technique before ATHIX can calculate a comparison score.
                    </p>
                  </div>
                )}
              </div>

              {/* GENERATED FILES */}

              <div style={{ marginTop: "30px" }}>
                <span className="overline">
                  GENERATED ANALYSIS FILES
                </span>

                <div
                  style={{
                    display: "grid",
                    gap: "10px",
                    marginTop: "14px",
                  }}
                >
                  {[
                    ["Pose landmarks", uploadResult.landmarks_file],
                    ["Joint angles", uploadResult.joint_angles_file],
                    ["Smoothed angles", uploadResult.smoothed_angles_file],
                    ["Movement phases", uploadResult.movement_phases_file],
                  ].map(([label, filename]) => (
                    <div
                      key={label}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: "12px",
                        padding: "12px",
                        background: "#171717",
                        borderRadius: "6px",
                        flexWrap: "wrap",
                      }}
                    >
                      <span style={{ color: "#cccccc" }}>
                        {label}
                      </span>

                      <span
                        style={{
                          color: "#999999",
                          fontSize: "12px",
                          overflowWrap: "anywhere",
                        }}
                      >
                        {filename || "Not available"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <p
                className="upload-pending-note"
                style={{ marginTop: "24px" }}
              >
                ATHIX currently provides an experimental comparison against a saved trainer video.
                This is not yet a coach-validated batting technique score.
              </p>

            </div>
          )}

          {/* RECORDING GUIDELINES */}

          <div className="upload-guidelines">

            <div className="guidelines-heading">
              <div>
                <span className="overline">
                  BEFORE YOU CONTINUE
                </span>

                <h2>Recording guidelines</h2>
              </div>

              <span className="guideline-count">
                03 POINTS
              </span>
            </div>

            <div className="guideline-grid">
              <div>
                <span>01</span>
                <h3>Full body visibility</h3>

                <p>
                  Keep your entire body visible throughout the shot.
                </p>
              </div>

              <div>
                <span>02</span>
                <h3>Stable camera</h3>

                <p>
                  Keep your camera steady while recording.
                </p>
              </div>

              <div>
                <span>03</span>
                <h3>Clear movement</h3>

                <p>
                  Make sure the bat and both feet are visible.
                </p>
              </div>
            </div>
          </div>

          {/* ANALYSIS BUTTON */}

          <button
            className="primary-button analyse-button"
            disabled={!videoFile || isUploading}
            onClick={handleAnalysis}
          >
            <span>
              {isUploading
                ? "Analysing your video..."
                : uploadResult
                  ? "Analyse again"
                  : videoFile
                    ? "Analyse my technique"
                    : "Select a video to continue"}
            </span>

            <span>
              {isUploading ? "..." : "→"}
            </span>
          </button>

          <p className="upload-disclaimer">
            Maximum file size: 20 MB. ATHIX currently extracts
            pose landmarks, calculates joint angles and identifies
            provisional movement phases.
          </p>

        </section>
      )}

      {/* FOOTER */}

      <footer>
        <span>ATHIX © 2026</span>
        <span>ENGINEERED FOR BETTER PERFORMANCE.</span>
      </footer>

    </main>
  );
}

export default App;
