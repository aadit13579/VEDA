// ── VEDA Renderer ──
// Upload → Voice/Button start → Pipeline → Display results → Read aloud (TTS)

export {} // make this file a module so declare global works


// ── Types ──
interface PipelineRegion {
  label: string
  bbox: number[]
  text?: string
  gemini_response?: string
  reading_order?: number
}

interface PipelinePage {
  page: number
  regions: PipelineRegion[]
}

interface PipelineResult {
  status: string
  file_id: string
  filename: string
  category: string
  total_pages: number
  start_page: number
  pages_processed: number
  final_document: {
    file_id: string
    total_pages: number
    pages: PipelinePage[]
  }
  steps: Array<{
    step: number
    name: string
    status: string
    time_ms: number
    error?: string
  }>
  total_time_ms: number
}

// ── State ──
interface VedaState {
  filePath: string | null
  fileName: string | null
  startPage: number
  isProcessing: boolean
  isListening: boolean
  pipelineResult: PipelineResult | null
  // TTS
  ttsUtterances: SpeechSynthesisUtterance[]
  ttsCurrentIndex: number
  ttsIsPaused: boolean
  ttsIsSpeaking: boolean
  // Audio recording
  mediaRecorder: MediaRecorder | null
  audioChunks: Blob[]
  // Keep-alive timer
  ttsKeepAliveTimer: number | null
}

const state: VedaState = {
  filePath: null,
  fileName: null,
  startPage: 1,
  isProcessing: false,
  isListening: false,
  pipelineResult: null,
  ttsUtterances: [],
  ttsCurrentIndex: 0,
  ttsIsPaused: false,
  ttsIsSpeaking: false,
  mediaRecorder: null,
  audioChunks: [],
  ttsKeepAliveTimer: null
}

// ── Gemini labels (match backend) ──
const GEMINI_LABELS = new Set([
  'figure',
  'table',
  'image',
  'picture',
  'isolate_formula',
  'figure_caption',
  'table_caption'
])

// ── Init ──
function init(): void {
  window.addEventListener('DOMContentLoaded', () => {
    setupUpload()
    setupVoice()
    setupStart()
    setupTTS()
    checkBackendHealth()
  })
}

// ── Backend Health Check ──
async function checkBackendHealth(): Promise<void> {
  try {
    const isHealthy = await window.electron.ipcRenderer.invoke('check-backend')
    if (isHealthy) {
      setStatus('✓ Backend connected — upload a document to begin.', 'success')
    } else {
      setStatus('⚠ Backend not running. Start with: uvicorn main:app --reload', 'warning')
    }
  } catch {
    setStatus('⚠ Backend not running. Start with: uvicorn main:app --reload', 'warning')
  }
}

// ══════════════════════════════════════
//  UPLOAD
// ══════════════════════════════════════
function setupUpload(): void {
  const uploadZone = document.getElementById('uploadZone')!

  uploadZone.addEventListener('click', async () => {
    if (state.isProcessing) return

    try {
      const filePaths: string[] = await window.electron.ipcRenderer.invoke('open-file-dialog')
      if (filePaths && filePaths.length > 0) {
        state.filePath = filePaths[0]
        state.fileName = filePaths[0].split(/[\\/]/).pop() || filePaths[0]

        // Show file name badge
        const fileBadge = document.getElementById('fileBadge')!
        const fileNameText = document.getElementById('fileNameText')!
        fileNameText.textContent = state.fileName
        fileBadge.classList.remove('hidden')

        // Show options row
        document.getElementById('optionsRow')!.classList.remove('hidden')

        // Hide results from a previous run
        document.getElementById('phaseResults')!.classList.add('hidden')
        window.speechSynthesis.cancel()

        setStatus(`"${state.fileName}" selected — choose how to start.`, 'info')
      }
    } catch (err) {
      console.error('File dialog error:', err)
      setStatus('Error opening file dialog.', 'error')
    }
  })
}

// ══════════════════════════════════════
//  VOICE COMMAND  (Web Speech API)
// ══════════════════════════════════════
function setupVoice(): void {
  const voiceBtn = document.getElementById('voiceBtn')!

  voiceBtn.addEventListener('click', async () => {
    if (state.isProcessing) return

    // If already recording, stop it
    if (state.isListening) {
      if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        state.mediaRecorder.stop()
      }
      return
    }

    // Start recording via MediaRecorder
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      // Find a supported MIME type
      let mimeType = 'audio/webm;codecs=opus'
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm'
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = '' // browser default
        }
      }

      const mediaRecorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined
      )

      state.mediaRecorder = mediaRecorder
      state.audioChunks = []
      state.isListening = true
      updateVoiceUI(true)

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          state.audioChunks.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        // Release the microphone
        stream.getTracks().forEach((track) => track.stop())

        state.isListening = false
        updateVoiceUI(false)

        // Combine chunks into a single blob
        const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' })

        if (audioBlob.size === 0) {
          setStatus('No audio recorded. Try again.', 'warning')
          return
        }

        setStatus('🔄 Transcribing your voice command…', 'info')

        try {
          // Convert WebM to WAV (SpeechRecognition backend needs WAV)
          const wavBuffer = await convertBlobToWav(audioBlob)

          // Send WAV audio to backend for transcription via IPC
          const transcript: string = await window.electron.ipcRenderer.invoke(
            'transcribe-audio',
            wavBuffer
          )

          console.log('Voice transcript:', transcript)

          const voiceFeedback = document.getElementById('voiceFeedback')!
          const voiceTranscript = document.getElementById('voiceTranscript')!
          const parsedPage = document.getElementById('parsedPage')!

          voiceFeedback.classList.remove('hidden')
          voiceTranscript.textContent = `"${transcript}"`

          // Parse page number from transcript
          const pageNum = parsePageNumber(transcript.toLowerCase())
          if (pageNum !== null) {
            state.startPage = pageNum
            parsedPage.textContent = `→ Starting from page ${pageNum}`
            setStatus(
              `Voice captured: starting from page ${pageNum}. Processing…`,
              'info'
            )
            // Auto-start pipeline
            runPipeline()
          } else {
            parsedPage.textContent =
              '→ Could not detect a page number. Try again.'
            setStatus(
              'Could not detect a page number. Try saying "go to page 3".',
              'warning'
            )
          }
        } catch (err) {
          console.error('Transcription error:', err)
          setStatus('Transcription failed. Try again.', 'error')
        }
      }

      // Start recording
      mediaRecorder.start()
      setStatus(
        '🎙 Recording… click again to stop, or say "go to page 5"',
        'info'
      )

      // Auto-stop after 8 seconds to prevent endless recording
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop()
        }
      }, 8000)
    } catch (err) {
      console.error('Microphone access error:', err)
      setStatus('⚠ Could not access microphone. Check permissions.', 'error')
      state.isListening = false
      updateVoiceUI(false)
    }
  })
}

function updateVoiceUI(recording: boolean): void {
  const voiceBtn = document.getElementById('voiceBtn')!
  const voicePulse = document.getElementById('voicePulse')!
  const voiceLabel = document.getElementById('voiceLabel')!

  if (recording) {
    voiceBtn.classList.add('border-purple-500', 'bg-purple-500/10')
    voicePulse.classList.remove('hidden')
    voiceLabel.textContent = 'Recording…'
    voiceLabel.classList.add('text-purple-400')
  } else {
    voiceBtn.classList.remove('border-purple-500', 'bg-purple-500/10')
    voicePulse.classList.add('hidden')
    voiceLabel.textContent = 'Speak Destination'
    voiceLabel.classList.remove('text-purple-400')
  }
}

function parsePageNumber(transcript: string): number | null {
  // Word → digit mapping
  const wordToNum: Record<string, number> = {
    one: 1,
    two: 2,
    three: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8,
    nine: 9,
    ten: 10,
    eleven: 11,
    twelve: 12,
    thirteen: 13,
    fourteen: 14,
    fifteen: 15,
    sixteen: 16,
    seventeen: 17,
    eighteen: 18,
    nineteen: 19,
    twenty: 20
  }

  // Try digit match: "page 5", "page 12"
  const digitMatch = transcript.match(/page\s+(\d+)/)
  if (digitMatch) {
    return parseInt(digitMatch[1], 10)
  }

  // Try word match: "page five", "page three"
  const wordMatch = transcript.match(/page\s+([a-z]+)/)
  if (wordMatch && wordToNum[wordMatch[1]]) {
    return wordToNum[wordMatch[1]]
  }

  // Try just a number in the string
  const anyDigit = transcript.match(/(\d+)/)
  if (anyDigit) {
    return parseInt(anyDigit[1], 10)
  }

  return null
}

// ══════════════════════════════════════
//  START BUTTON  (Page 1)
// ══════════════════════════════════════
function setupStart(): void {
  const startBtn = document.getElementById('startBtn')!

  startBtn.addEventListener('click', () => {
    if (!state.filePath) {
      setStatus('Please upload a document first.', 'warning')
      return
    }
    if (state.isProcessing) return

    state.startPage = 1
    runPipeline()
  })
}

// ══════════════════════════════════════
//  RUN PIPELINE
// ══════════════════════════════════════
async function runPipeline(): Promise<void> {
  if (!state.filePath || state.isProcessing) return

  state.isProcessing = true
  window.speechSynthesis.cancel()

  const startBtn = document.getElementById('startBtn')!
  const voiceBtn = document.getElementById('voiceBtn')!
  startBtn.setAttribute('disabled', 'true')
  voiceBtn.setAttribute('disabled', 'true')
  showProgress(true)

  // Start indeterminate progress bar + elapsed timer
  const progressFill = document.getElementById('progressFill')!
  progressFill.classList.add('progress-indeterminate')
  const startTime = Date.now()
  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0)
    setProgressText(
      `⏳ Processing "${state.fileName}" from page ${state.startPage}… (${elapsed}s elapsed)`
    )
  }, 1000)
  setProgressText(`⏳ Processing "${state.fileName}" from page ${state.startPage}…`)
  setStatus('Backend is running: Upload → Layout → Sort → OCR → Finalize', 'info')

  try {
    const result: PipelineResult = await window.electron.ipcRenderer.invoke(
      'run-pipeline',
      state.filePath,
      state.startPage
    )

    // Stop timer, switch to determinate bar
    clearInterval(timerInterval)
    progressFill.classList.remove('progress-indeterminate')

    state.pipelineResult = result
    setProgress(100, `✓ Done in ${(result.total_time_ms / 1000).toFixed(1)}s`)
    setStatus(
      `✓ Processed ${result.pages_processed} page(s) — reading aloud from page ${result.start_page}.`,
      'success'
    )

    // Log step details
    console.log('Pipeline result:', result)
    for (const step of result.steps) {
      console.log(`  Step ${step.step}: ${step.name} — ${step.status} (${step.time_ms}ms)`)
    }

    // Render results and start TTS
    renderResults(result)
    buildTTSQueue(result)
    startTTS()
  } catch (err) {
    clearInterval(timerInterval)
    progressFill.classList.remove('progress-indeterminate')

    const message = err instanceof Error ? err.message : String(err)
    setStatus(`✗ Pipeline failed: ${message}`, 'error')
    setProgress(0, 'Failed')
    console.error('Pipeline error:', err)
  } finally {
    state.isProcessing = false
    startBtn.removeAttribute('disabled')
    voiceBtn.removeAttribute('disabled')
  }
}

// ══════════════════════════════════════
//  RENDER RESULTS
// ══════════════════════════════════════
function renderResults(result: PipelineResult): void {
  const phaseResults = document.getElementById('phaseResults')!
  const resultsPanel = document.getElementById('resultsPanel')!

  phaseResults.classList.remove('hidden')
  resultsPanel.innerHTML = ''

  const pages = result.final_document?.pages || []

  if (pages.length === 0) {
    resultsPanel.innerHTML =
      '<p class="text-sm text-veda-text-muted text-center py-8">No content extracted.</p>'
    return
  }

  for (const page of pages) {
    const pageDiv = document.createElement('div')
    pageDiv.className = 'page-block'

    const title = document.createElement('div')
    title.className = 'page-title'
    title.textContent = `Page ${page.page}`
    pageDiv.appendChild(title)

    const sorted = [...page.regions].sort(
      (a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)
    )

    for (const region of sorted) {
      const text = region.text || region.gemini_response || ''
      if (!text.trim()) continue

      const label = (region.label || '').toLowerCase().replace(' ', '_')
      const isGemini = GEMINI_LABELS.has(label)

      const block = document.createElement('div')
      block.className = 'region-block'
      block.id = `region-${page.page}-${region.reading_order ?? 0}`

      const labelEl = document.createElement('div')
      labelEl.className = `region-label ${isGemini ? 'gemini' : ''}`
      labelEl.textContent = isGemini ? `${region.label} (Gemini)` : region.label || 'text'
      block.appendChild(labelEl)

      const textEl = document.createElement('div')
      textEl.className = 'region-text'
      textEl.textContent = text.trim()
      block.appendChild(textEl)

      pageDiv.appendChild(block)
    }

    resultsPanel.appendChild(pageDiv)
  }
}

// ══════════════════════════════════════
//  TEXT-TO-SPEECH  (TTS)
// ══════════════════════════════════════
function setupTTS(): void {
  document.getElementById('ttsPlayBtn')!.addEventListener('click', () => {
    if (state.ttsIsPaused) {
      resumeTTS()
    } else if (!state.ttsIsSpeaking) {
      startTTS()
    }
  })

  document.getElementById('ttsPauseBtn')!.addEventListener('click', () => {
    pauseTTS()
  })

  document.getElementById('ttsResumeBtn')!.addEventListener('click', () => {
    resumeTTS()
  })

  document.getElementById('ttsStopBtn')!.addEventListener('click', () => {
    stopTTS()
  })
}

function buildTTSQueue(result: PipelineResult): void {
  state.ttsUtterances = []
  state.ttsCurrentIndex = 0

  const pages = result.final_document?.pages || []

  for (const page of pages) {
    // Page announcement
    const pageAnnounce = new SpeechSynthesisUtterance(`Page ${page.page}.`)
    pageAnnounce.rate = 1.0
    state.ttsUtterances.push(pageAnnounce)

    const sorted = [...page.regions].sort(
      (a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)
    )

    for (const region of sorted) {
      const text = region.text || region.gemini_response || ''
      if (!text.trim()) continue

      const label = (region.label || '').toLowerCase().replace(' ', '_')
      const isGemini = GEMINI_LABELS.has(label)

      // For Gemini regions, prefix with a short label for context
      let speakText = text.trim()
      if (isGemini) {
        speakText = `${region.label} description: ${speakText}`
      }

      // Split long text into chunks (speechSynthesis can fail on very long strings)
      // 30 words is safe to stay under the 15-second Chrome cutoff
      const chunks = splitTextForTTS(speakText, 30)

      for (const chunk of chunks) {
        const utterance = new SpeechSynthesisUtterance(chunk)
        utterance.rate = 1.0
        utterance.lang = 'en-US'

        // Store metadata for highlighting
        const regionId = `region-${page.page}-${region.reading_order ?? 0}`
        utterance.addEventListener('start', () => highlightRegion(regionId, true))
        utterance.addEventListener('end', () => highlightRegion(regionId, false))

        state.ttsUtterances.push(utterance)
      }
    }
  }
}

function splitTextForTTS(text: string, maxWords: number): string[] {
  const words = text.split(/\s+/)
  const chunks: string[] = []
  for (let i = 0; i < words.length; i += maxWords) {
    chunks.push(words.slice(i, i + maxWords).join(' '))
  }
  return chunks
}

function startTTS(): void {
  if (state.ttsUtterances.length === 0) return

  window.speechSynthesis.cancel()
  state.ttsCurrentIndex = 0
  state.ttsIsSpeaking = true
  state.ttsIsPaused = false
  updateTTSUI('playing')
  
  // Start Keep-Alive to prevent Chrome from silent halting after 15s
  if (state.ttsKeepAliveTimer) window.clearInterval(state.ttsKeepAliveTimer)
  state.ttsKeepAliveTimer = window.setInterval(() => {
    if (state.ttsIsSpeaking && !state.ttsIsPaused) {
      window.speechSynthesis.pause()
      window.speechSynthesis.resume()
    }
  }, 10000)

  speakNext()
}

function speakNext(): void {
  if (state.ttsCurrentIndex >= state.ttsUtterances.length) {
    state.ttsIsSpeaking = false
    updateTTSUI('stopped')
    setTTSStatus('Finished reading.')
    return
  }

  const utterance = state.ttsUtterances[state.ttsCurrentIndex]

  utterance.onend = () => {
    state.ttsCurrentIndex++
    if (state.ttsIsSpeaking && !state.ttsIsPaused) {
      speakNext()
    }
  }

  utterance.onerror = (event) => {
    console.error('TTS error:', event)
    state.ttsCurrentIndex++
    if (state.ttsIsSpeaking) speakNext()
  }

  setTTSStatus(`Speaking (${state.ttsCurrentIndex + 1}/${state.ttsUtterances.length})`)
  window.speechSynthesis.speak(utterance)
}

function pauseTTS(): void {
  window.speechSynthesis.pause()
  state.ttsIsPaused = true
  updateTTSUI('paused')
  setTTSStatus('Paused')
}

function resumeTTS(): void {
  window.speechSynthesis.resume()
  state.ttsIsPaused = false
  updateTTSUI('playing')
  setTTSStatus(`Speaking (${state.ttsCurrentIndex + 1}/${state.ttsUtterances.length})`)
}

function stopTTS(): void {
  window.speechSynthesis.cancel()
  state.ttsIsSpeaking = false
  state.ttsIsPaused = false
  state.ttsCurrentIndex = 0
  if (state.ttsKeepAliveTimer) {
    window.clearInterval(state.ttsKeepAliveTimer)
    state.ttsKeepAliveTimer = null
  }
  updateTTSUI('stopped')
  setTTSStatus('Stopped')

  // Remove all highlights
  document.querySelectorAll('.region-block.speaking').forEach((el) => {
    el.classList.remove('speaking')
  })
}

function highlightRegion(regionId: string, active: boolean): void {
  const el = document.getElementById(regionId)
  if (!el) return

  if (active) {
    // Remove previous highlights
    document.querySelectorAll('.region-block.speaking').forEach((other) => {
      other.classList.remove('speaking')
    })
    el.classList.add('speaking')
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  } else {
    el.classList.remove('speaking')
  }
}

function updateTTSUI(mode: 'playing' | 'paused' | 'stopped'): void {
  const playBtn = document.getElementById('ttsPlayBtn')!
  const pauseBtn = document.getElementById('ttsPauseBtn')!
  const resumeBtn = document.getElementById('ttsResumeBtn')!
  const visualizer = document.getElementById('ttsVisualizer')

  playBtn.classList.add('hidden')
  pauseBtn.classList.add('hidden')
  resumeBtn.classList.add('hidden')

  if (visualizer) visualizer.classList.remove('active')

  switch (mode) {
    case 'playing':
      pauseBtn.classList.remove('hidden')
      if (visualizer) visualizer.classList.add('active')
      break
    case 'paused':
      resumeBtn.classList.remove('hidden')
      break
    case 'stopped':
      playBtn.classList.remove('hidden')
      break
  }
}

function setTTSStatus(msg: string): void {
  const el = document.getElementById('ttsStatus')
  if (el) el.textContent = msg
}

// ══════════════════════════════════════
//  PROGRESS / STATUS HELPERS
// ══════════════════════════════════════
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

function setProgressText(label: string): void {
  const text = document.getElementById('progressText')!
  text.textContent = label
}

function setStatus(msg: string, type: 'info' | 'success' | 'warning' | 'error' = 'info'): void {
  const el = document.getElementById('statusText')
  if (!el) return

  el.textContent = msg
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
// ══════════════════════════════════════
//  AUDIO CONVERSION  (WebM → WAV)
// ══════════════════════════════════════
async function convertBlobToWav(blob: Blob): Promise<ArrayBuffer> {
  // Decode the WebM audio using AudioContext
  const audioCtx = new AudioContext({ sampleRate: 16000 })
  const arrayBuffer = await blob.arrayBuffer()
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)

  // Get mono channel data
  const channelData = audioBuffer.getChannelData(0)
  const wavBuffer = encodeWav(channelData, audioBuffer.sampleRate)

  audioCtx.close()
  return wavBuffer
}

function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1
  const bitsPerSample = 16
  const bytesPerSample = bitsPerSample / 8
  const blockAlign = numChannels * bytesPerSample
  const dataSize = samples.length * bytesPerSample
  const headerSize = 44
  const totalSize = headerSize + dataSize

  const buffer = new ArrayBuffer(totalSize)
  const view = new DataView(buffer)

  // RIFF header
  writeAscii(view, 0, 'RIFF')
  view.setUint32(4, totalSize - 8, true)
  writeAscii(view, 8, 'WAVE')

  // fmt sub-chunk
  writeAscii(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // sub-chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * blockAlign, true) // byte rate
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)

  // data sub-chunk
  writeAscii(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  // Convert float32 samples [-1.0, 1.0] to int16
  let offset = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    const val = s < 0 ? s * 0x8000 : s * 0x7fff
    view.setInt16(offset, val, true)
    offset += 2
  }

  return buffer
}

function writeAscii(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}

init()
