
import { useEffect, useRef, useState } from 'react'

const API_URL = 'http://127.0.0.1:8000'
const CAPTURE_INTERVAL = 700

function WebcamPanel() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const busyRef = useRef(false)

  const [active, setActive] = useState(false)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [processedFrames, setProcessedFrames] = useState(0)

  const startCamera = async () => {
    setError('')
    setResult(null)
    setProcessedFrames(0)
    setStarting(true)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'environment',
        },
        audio: false,
      })

      streamRef.current = stream
      setActive(true)
    } catch (err) {
      console.error(err)
      setError(
        'Unable to access the camera. Please allow camera permission and try again.'
      )
    } finally {
      setStarting(false)
    }
  }

  const stopCamera = () => {
    setActive(false)

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    busyRef.current = false
  }

  useEffect(() => {
    if (!active) return

    let intervalId
    let cancelled = false

    const initializeCamera = async () => {
      const video = videoRef.current

      if (!video || !streamRef.current) return

      video.srcObject = streamRef.current

      try {
        await video.play()
      } catch (err) {
        console.error('Camera playback error:', err)
        setError('Camera preview could not be started.')
        stopCamera()
        return
      }

      const analyzeFrame = async () => {
        if (cancelled || busyRef.current) return

        const currentVideo = videoRef.current
        const canvas = canvasRef.current

        if (
          !currentVideo ||
          !canvas ||
          currentVideo.readyState < 2 ||
          currentVideo.videoWidth === 0
        ) {
          return
        }

        busyRef.current = true

        try {
          canvas.width = currentVideo.videoWidth
          canvas.height = currentVideo.videoHeight

          const context = canvas.getContext('2d')
          context.drawImage(
            currentVideo,
            0,
            0,
            canvas.width,
            canvas.height
          )

          const blob = await new Promise((resolve) => {
            canvas.toBlob(resolve, 'image/jpeg', 0.75)
          })

          if (!blob || cancelled) return

          const formData = new FormData()
          formData.append('file', blob, 'webcam-frame.jpg')

          const response = await fetch(`${API_URL}/predict/visualize`, {
            method: 'POST',
            body: formData,
          })

          if (!response.ok) {
            throw new Error(`API Error: ${response.status}`)
          }

          const data = await response.json()

          if (!cancelled) {
            setResult(data)
            setProcessedFrames((count) => count + 1)
            setError('')
          }
        } catch (err) {
          console.error('Webcam detection error:', err)

          if (!cancelled) {
            setError('Detection API error. Check that FastAPI is running.')
          }
        } finally {
          busyRef.current = false
        }
      }

      intervalId = window.setInterval(
        analyzeFrame,
        CAPTURE_INTERVAL
      )

      analyzeFrame()
    }

    initializeCamera()

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
      busyRef.current = false
    }
  }, [active])

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  const detections = result?.detections || []

  return (
    <section className="panel webcam-panel">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">LIVE COMPUTER VISION</span>
          <h3>Real-Time Webcam Detection</h3>
        </div>

        <span className="panel-number">03</span>
      </div>

      <div className="webcam-toolbar">
        <div className="webcam-status">
          <span className={`webcam-dot ${active ? 'active' : ''}`} />
          {active ? 'Camera active' : 'Camera stopped'}
        </div>

        {!active ? (
          <button
            className="webcam-start"
            onClick={startCamera}
            disabled={starting}
          >
            {starting ? 'Starting camera...' : '▶ Start Camera'}
          </button>
        ) : (
          <button
            className="webcam-stop"
            onClick={stopCamera}
          >
            ■ Stop Camera
          </button>
        )}
      </div>

      <div className="webcam-screen">
        <video
          ref={videoRef}
          className="webcam-video"
          autoPlay
          muted
          playsInline
        />

        {result?.annotated_image && active && (
          <img
            className="webcam-annotations"
            src={`data:image/jpeg;base64,${result.annotated_image}`}
            alt="Live PPE detection"
          />
        )}

        {!active && (
          <div className="webcam-placeholder">
            <span>◉</span>
            <strong>Camera preview</strong>
            <p>Start the camera to begin live PPE monitoring.</p>
          </div>
        )}

        <canvas ref={canvasRef} hidden />
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="webcam-metrics">
        <div>
          <span>DETECTIONS</span>
          <strong>{active && result ? result.total_detections ?? detections.length : '—'}</strong>
        </div>

        <div>
          <span>INFERENCE TIME</span>
          <strong>
            {active && result
              ? `${Number(result.inference_time_ms || 0).toFixed(1)} ms`
              : '—'}
          </strong>
        </div>

        <div>
          <span>PROCESSED FRAMES</span>
          <strong>{processedFrames}</strong>
        </div>
      </div>

      {active && detections.length > 0 && (
        <div className="webcam-detection-list">
          <h4>Current frame detections</h4>

          <div className="webcam-tags">
            {detections.map((item, index) => {
              const name = item.class_name ?? item.class ?? item.label ?? 'Unknown'
              const confidence = Number(item.confidence ?? item.score ?? 0)

              return (
                <span
                  className={name.toLowerCase().startsWith('no_') ? 'webcam-tag risk' : 'webcam-tag'}
                  key={`${name}-${index}`}
                >
                  {name.replaceAll('_', ' ')} · {(confidence * 100).toFixed(0)}%
                </span>
              )
            })}
          </div>
        </div>
      )}

      <p className="webcam-note">
        Frames are captured locally and analyzed by the SafeVision AI API.
        Allow camera access when your browser asks for permission.
      </p>
    </section>
  )
}

export default WebcamPanel
