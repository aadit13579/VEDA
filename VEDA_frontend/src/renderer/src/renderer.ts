// ── VEDA Renderer ──
// Handles: file upload → backend pipeline, mic recording, progress tracking

interface UploadResult {
  status: string
  file_id: string
  filename: string
  detected_mime: string
  category: string
  process_time_ms: number
}

interface VedaState {
  uploadedFiles: string[]
  uploadResults: UploadResult[]
  audioBlob: Blob | null
  isRecording: boolean
  isProcessing: boolean
  mediaRecorder: MediaRecorder | null
  audioChunks: Blob[]
}

const state: VedaState = {
  uploadedFiles: [],
  uploadResults: [],
  audioBlob: null,
  isRecording: false,
  isProcessing: false,
  mediaRecorder: null,
  audioChunks: []
}

function init(): void {
  window.addEventListener('DOMContentLoaded', () => {
    setupUpload()
    setupMic()
    setupStart()
    checkBackendHealth()
  })
}

// ── Backend Health Check ──
async function checkBackendHealth(): Promise<void> {
  try {
    const isHealthy = await window.electron.ipcRenderer.invoke('check-backend')
    if (isHealthy) {
      setStatus('✓ Backend connected — ready to process.', 'success')
    } else {
      setStatus('⚠ Backend not running. Start it with: uvicorn main:app --reload', 'warning')
    }
  } catch {
    setStatus('⚠ Backend not running. Start it with: uvicorn main:app --reload', 'warning')
  }
}

// ── Upload Zone ──
function setupUpload(): void {
  const uploadZone = document.getElementById('uploadZone')!

  uploadZone.addEventListener('click', async () => {
    if (state.isProcessing) return

    try {
      const filePaths: string[] = await window.electron.ipcRenderer.invoke('open-file-dialog')
      if (filePaths && filePaths.length > 0) {
        state.uploadedFiles = filePaths
        state.uploadResults = []
        renderFileList(filePaths)
        setStatus(`${filePaths.length} file(s) selected — click Start to process.`, 'info')
      }
    } catch (err) {
      console.error('File dialog error:', err)
      setStatus('Error opening file dialog.', 'error')
    }
  })
}

function renderFileList(filePaths: string[]): void {
  const fileList = document.getElementById('fileList')!
  fileList.innerHTML = ''
  fileList.classList.remove('hidden')

  filePaths.forEach((fp) => {
    const fileName = fp.split(/[\\/]/).pop() || fp
    const item = document.createElement('div')
    item.className =
      'flex items-center gap-2 bg-veda-surface/80 rounded-lg px-3 py-2 text-sm text-veda-text-muted'
    item.innerHTML = `
      <svg class="w-4 h-4 text-veda-accent shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
      </svg>
      <span class="truncate">${fileName}</span>
      <span class="ml-auto text-xs text-veda-text-muted/40" id="file-status-${filePaths.indexOf(fp)}">ready</span>
    `
    fileList.appendChild(item)
  })
}

function updateFileStatus(index: number, status: string, color: string): void {
  const el = document.getElementById(`file-status-${index}`)
  if (el) {
    el.textContent = status
    el.className = `ml-auto text-xs ${color}`
  }
}

// ── Mic Recording ──
function setupMic(): void {
  const micBtn = document.getElementById('micBtn')!

  micBtn.addEventListener('click', async () => {
    if (state.isProcessing) return

    if (state.isRecording) {
      stopRecording()
    } else {
      await startRecording()
    }
  })
}

async function startRecording(): Promise<void> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    state.mediaRecorder = new MediaRecorder(stream)
    state.audioChunks = []

    state.mediaRecorder.ondataavailable = (e: BlobEvent) => {
      if (e.data.size > 0) {
        state.audioChunks.push(e.data)
      }
    }

    state.mediaRecorder.onstop = () => {
      state.audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' })
      state.audioChunks = []
      stream.getTracks().forEach((t) => t.stop())
      setStatus('Voice recorded ✓ — click Start to process.', 'success')
    }

    state.mediaRecorder.start()
    state.isRecording = true
    updateMicUI(true)
    setStatus('🎙 Recording... click mic again to stop.', 'info')
  } catch (err) {
    console.error('Mic error:', err)
    setStatus('Microphone access denied.', 'error')
  }
}

function stopRecording(): void {
  if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
    state.mediaRecorder.stop()
  }
  state.isRecording = false
  updateMicUI(false)
}

function updateMicUI(recording: boolean): void {
  const micBtn = document.getElementById('micBtn')!
  const micIcon = document.getElementById('micIcon')!
  const micPulse = document.getElementById('micPulse')!
  const micLabel = document.getElementById('micLabel')!

  if (recording) {
    micBtn.classList.add('border-red-500', 'bg-red-500/10')
    micBtn.classList.remove('border-veda-border', 'bg-veda-surface')
    micIcon.classList.add('text-red-400')
    micIcon.classList.remove('text-veda-text-muted')
    micPulse.classList.remove('hidden')
    micLabel.textContent = 'Recording...'
    micLabel.classList.add('text-red-400')
    micLabel.classList.remove('text-veda-text-muted')
  } else {
    micBtn.classList.remove('border-red-500', 'bg-red-500/10')
    micBtn.classList.add('border-veda-border', 'bg-veda-surface')
    micIcon.classList.remove('text-red-400')
    micIcon.classList.add('text-veda-text-muted')
    micPulse.classList.add('hidden')
    micLabel.textContent = 'Record Voice'
    micLabel.classList.remove('text-red-400')
    micLabel.classList.add('text-veda-text-muted')
  }
}

// ── Start Button — Full Pipeline ──
function setupStart(): void {
  const startBtn = document.getElementById('startBtn')!

  startBtn.addEventListener('click', async () => {
    if (state.uploadedFiles.length === 0) {
      setStatus('Please upload at least one document first.', 'warning')
      return
    }

    if (state.isProcessing) return

    state.isProcessing = true
    startBtn.setAttribute('disabled', 'true')
    showProgress(true)

    try {
      // ── STEP 1: Upload each file to backend ──
      setProgress(0, `Uploading ${state.uploadedFiles.length} file(s)...`)

      for (let i = 0; i < state.uploadedFiles.length; i++) {
        const filePath = state.uploadedFiles[i]
        const fileName = filePath.split(/[\\/]/).pop() || filePath

        updateFileStatus(i, 'uploading...', 'text-yellow-400')
        setProgress(
          ((i) / state.uploadedFiles.length) * 33,
          `Uploading: ${fileName}`
        )

        try {
          const result: UploadResult = await window.electron.ipcRenderer.invoke('upload-file', filePath)
          state.uploadResults.push(result)
          updateFileStatus(i, `✓ ${result.category}`, 'text-green-400')
        } catch (err) {
          updateFileStatus(i, '✗ failed', 'text-red-400')
          console.error(`Upload failed for ${fileName}:`, err)
          throw new Error(`Failed to upload ${fileName}`)
        }
      }

      setProgress(33, 'All files uploaded ✓')

      // ── STEP 2: Layout Analysis for each file ──
      for (let i = 0; i < state.uploadResults.length; i++) {
        const result = state.uploadResults[i]
        setProgress(
          33 + ((i) / state.uploadResults.length) * 33,
          `Analyzing layout: ${result.filename}`
        )

        try {
          await window.electron.ipcRenderer.invoke('analyze-layout', result.file_id)
        } catch (err) {
          console.error(`Layout analysis failed for ${result.filename}:`, err)
          throw new Error(`Layout analysis failed for ${result.filename}`)
        }
      }

      setProgress(66, 'Layout analysis complete ✓')

      // ── STEP 3: Spatial Sort for each file ──
      for (let i = 0; i < state.uploadResults.length; i++) {
        const result = state.uploadResults[i]
        setProgress(
          66 + ((i) / state.uploadResults.length) * 34,
          `Sorting reading order: ${result.filename}`
        )

        try {
          await window.electron.ipcRenderer.invoke('spatial-sort', result.file_id)
        } catch (err) {
          console.error(`Spatial sort failed for ${result.filename}:`, err)
          throw new Error(`Spatial sort failed for ${result.filename}`)
        }
      }

      setProgress(100, 'Processing complete! ✓')
      setStatus('✓ All files processed successfully!', 'success')

      // Log results for debugging
      console.log('Pipeline results:', state.uploadResults)

    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setStatus(`✗ Error: ${message}`, 'error')
      console.error('Pipeline error:', err)
    } finally {
      state.isProcessing = false
      startBtn.removeAttribute('disabled')
    }
  })
}

// ── Progress Bar ──
function showProgress(visible: boolean): void {
  const bar = document.getElementById('progressContainer')!
  if (visible) {
    bar.classList.remove('hidden')
  } else {
    bar.classList.add('hidden')
  }
}

function setProgress(percent: number, label: string): void {
  const fill = document.getElementById('progressFill')!
  const text = document.getElementById('progressText')!
  fill.style.width = `${Math.min(100, Math.max(0, percent))}%`
  text.textContent = label
}

// ── Helpers ──
function setStatus(msg: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
  const el = document.getElementById('statusText')
  if (!el) return

  el.textContent = msg

  // Reset classes
  el.className = 'text-sm'

  switch (type) {
    case 'success':
      el.classList.add('text-green-400')
      break
    case 'warning':
      el.classList.add('text-yellow-400')
      break
    case 'error':
      el.classList.add('text-red-400')
      break
    default:
      el.classList.add('text-veda-text-muted/60')
  }
}

init()
