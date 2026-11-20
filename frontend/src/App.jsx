
import { useRef, useState } from 'react'
import './App.css'
import WebcamPanel from './WebcamPanel.jsx'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const fileInputRef = useRef(null)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)

  const detections = result?.detections || []

  const handleFile = (selectedFile) => {
    if (!selectedFile) return

    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file.')
      return
    }

    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setError('')
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an image first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${API_URL}/predict/visualize`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `API Error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(
        'Cannot connect to the API. Make sure FastAPI is running on port 8000.',
      )
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const getClassName = (item) =>
    item.class_name ?? item.class ?? item.label ?? 'Unknown'

  const getConfidence = (item) => {
    const value = item.confidence ?? item.score ?? 0
    return `${(Number(value) * 100).toFixed(1)}%`
  }

  const getConfidenceValue = (item) =>
    Number(item.confidence ?? item.score ?? 0)

  const formatTime = (value) =>
    Number(value ?? 0).toFixed(2)

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">🦺</div>
          <div>
            <h2>SafeVision</h2>
            <span>AI Monitoring System</span>
          </div>
        </div>

        <div className="sidebar-label">MAIN MENU</div>

        <nav className="navigation">
          <div className="nav-item active">
            <span>▦</span> Dashboard
          </div>
          <div className="nav-item">
            <span>◉</span> Detection Studio
          </div>
          <div className="nav-item">
            <span>▤</span> Detection History
          </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="online-dot" />
            <div>
              <strong>AI System</strong>
              <p>Ready for detection</p>
            </div>
          </div>
          <div className="version">SMART PPE DETECTION · V1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">Workspace / Dashboard</div>
            <h1>Safety Intelligence Dashboard</h1>
            <p className="page-subtitle">
              Real-time Personal Protective Equipment monitoring
            </p>
          </div>

          <div className="topbar-right">
            <div className="status-pill">
              <span className="online-dot" /> System Online
            </div>
            <div className="avatar">AI</div>
          </div>
        </header>

        <section className="welcome-banner">
          <div>
            <span className="eyebrow">COMPUTER VISION · YOLO11</span>
            <h2>Protecting people with intelligent vision.</h2>
            <p>
              Analyze workplace images and identify PPE equipment,
              missing protection and potential safety risks.
            </p>
          </div>
          <div className="banner-art">
            <div className="art-circle">🦺</div>
            <div className="art-orbit orbit-one" />
            <div className="art-orbit orbit-two" />
          </div>
        </section>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon purple">◉</div>
            <div className="stat-label">Total Detections</div>
            <div className="stat-value">
              {result ? result.total_detections ?? detections.length : '—'}
            </div>
            <div className="stat-note">Objects identified in image</div>
          </div>

          <div className="stat-card">
            <div className="stat-icon green">✓</div>
            <div className="stat-label">Model Status</div>
            <div className="stat-value status-text">Ready</div>
            <div className="stat-note">YOLO11n · ONNX Runtime</div>
          </div>

          <div className="stat-card">
            <div className="stat-icon blue">◷</div>
            <div className="stat-label">Inference Time</div>
            <div className="stat-value">
              {result ? `${formatTime(result.inference_time_ms)} ms` : '—'}
            </div>
            <div className="stat-note">Processing duration</div>
          </div>

          <div className="stat-card">
            <div className="stat-icon orange">◎</div>
            <div className="stat-label">Image Status</div>
            <div className="stat-value status-text">
              {result ? 'Analyzed' : 'Waiting'}
            </div>
            <div className="stat-note">Image analysis state</div>
          </div>
        </section>

        <section className="workspace-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">IMAGE ANALYSIS</span>
                <h3>Upload an image</h3>
              </div>
              <span className="panel-number">01</span>
            </div>

            <div
              className={`drop-zone ${dragging ? 'dragging' : ''}`}
              onDragOver={(event) => {
                event.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragging(false)
                handleFile(event.dataTransfer.files[0])
              }}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'Enter') fileInputRef.current?.click()
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(event) => handleFile(event.target.files[0])}
              />

              <div className="upload-icon">↑</div>
              <h4>Drop your image here</h4>
              <p>or click to browse files</p>
              <span className="file-types">PNG · JPG · JPEG · WEBP</span>
            </div>

            {file && (
              <div className="selected-file">
                <div className="file-icon">▧</div>
                <div className="file-info">
                  <strong>{file.name}</strong>
                  <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <button className="remove-file" onClick={reset}>×</button>
              </div>
            )}

            <button
              className="analyze-button"
              onClick={handleUpload}
              disabled={!file || loading}
            >
              {loading ? (
                <><span className="spinner" /> Analyzing image...</>
              ) : (
                <>✦ &nbsp; Run AI Detection</>
              )}
            </button>

            {error && <div className="error-message">{error}</div>}

            <div className="upload-tip">
              <span>ⓘ</span>
              Images are processed locally by your AI inference API.
            </div>
          </div>

          <div className="panel preview-panel">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">VISUAL INSPECTION</span>
                <h3>Detection preview</h3>
              </div>
              <span className="panel-number">02</span>
            </div>

            <div className="image-preview">
              {result?.annotated_image ? (
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image}`}
                  alt="PPE detection result"
                />
              ) : preview ? (
                <img src={preview} alt="Selected image preview" />
              ) : (
                <div className="empty-preview">
                  <div className="empty-icon">▧</div>
                  <strong>No image selected</strong>
                  <span>Your image analysis will appear here</span>
                </div>
              )}
            </div>

            <div className="preview-footer">
              <span>
                {result ? '✓ Analysis completed' : 'Waiting for an image'}
              </span>
              {result && (
                <span className="success-label">ANNOTATED RESULT</span>
              )}
            </div>
          </div>
        </section>
        <WebcamPanel />
        <section className="panel results-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">AI OUTPUT</span>
              <h3>Detected objects</h3>
            </div>
            <span className="results-count">
              {detections.length} OBJECTS
            </span>
          </div>

          {detections.length > 0 ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>CLASS</th>
                    <th>CONFIDENCE</th>
                    <th>CONFIDENCE LEVEL</th>
                    <th>STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {detections.map((item, index) => {
                    const confidence = getConfidenceValue(item)
                    const className = getClassName(item)
                    const isMissing = className.toLowerCase().startsWith('no_')

                    return (
                      <tr key={`${className}-${index}`}>
                        <td>
                          <div className="class-cell">
                            <span className={`class-dot ${isMissing ? 'warning' : ''}`} />
                            {className.replaceAll('_', ' ')}
                          </div>
                        </td>
                        <td className="confidence-value">
                          {getConfidence(item)}
                        </td>
                        <td>
                          <div className="confidence-track">
                            <div
                              className="confidence-fill"
                              style={{ width: `${Math.min(confidence * 100, 100)}%` }}
                            />
                          </div>
                        </td>
                        <td>
                          <span className={`result-badge ${isMissing ? 'risk' : 'detected'}`}>
                            {isMissing ? 'Missing PPE' : 'Detected'}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-results">
              <span>◎</span>
              <p>Detection results will be displayed here after analysis.</p>
            </div>
          )}
        </section>

        <footer className="footer">
          <span>SafeVision AI · Smart PPE Detection System</span>
          <span>Powered by YOLO11n + ONNX Runtime</span>
        </footer>
      </main>
    </div>
  )
}

export default App
