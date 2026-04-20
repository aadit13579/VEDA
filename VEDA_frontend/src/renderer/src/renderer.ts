// ── VEDA Renderer ──
// Upload → Voice/Button start → Pipeline (streaming) → Display results incrementally → Read aloud (TTS)

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
  debug_image_url?: string
}

interface StartPipelineResult {
  status: string
  file_id: string
  filename: string
  category: string
  total_pages: number
  start_page: number
}

interface PageReadyEvent {
  page: number
  total_pages: number
  pages_processed: number
  page_data: PipelinePage
  page_time_ms: number
}

interface PipelineCompleteEvent {
  file_id: string
  total_pages: number
  pages_processed: number
  output_path: string
  total_time_ms: number
  counters: {
    ocr: number
    gemini: number
    pymupdf: number
  }
}

interface PipelineErrorEvent {
  file_id: string
  error: string
  pages_processed: number
}

interface PipelineEvent {
  type: 'page_ready' | 'error' | 'complete'
  data: PageReadyEvent | PipelineCompleteEvent | PipelineErrorEvent
}

// ── State ──
interface VedaState {
  filePath: string | null
  fileName: string | null
  startPage: number
  isProcessing: boolean
  isListening: boolean
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
  // Pipeline event listener cleanup
  pipelineEventCleanup: (() => void) | null
}

const state: VedaState = {
  filePath: null,
  fileName: null,
  startPage: 1,
  isProcessing: false,
  isListening: false,
  ttsUtterances: [],
  ttsCurrentIndex: 0,
  ttsIsPaused: false,
  ttsIsSpeaking: false,
  mediaRecorder: null,
  audioChunks: [],
  ttsKeepAliveTimer: null,
  pipelineEventCleanup: null
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

// ── Theme Toggling ──
function setupTheme(): void {
  const toggleBtn = document.getElementById('themeToggleBtn')!
  
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const savedTheme = localStorage.getItem('veda-theme')
  const isDark = savedTheme === 'dark' || (!savedTheme && prefersDark)
  
  if (isDark) document.documentElement.setAttribute('data-theme', 'dark')
  
  toggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme')
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark'
    
    if (newTheme === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark')
    } else {
      document.documentElement.removeAttribute('data-theme')
    }
    localStorage.setItem('veda-theme', newTheme)
  })
}

// ── Floating Particles (Visual Effect) ──
function spawnParticles(): void {
  const canvas = document.getElementById('particleCanvas')
  if (!canvas) return

  const count = 30
  for (let i = 0; i < count; i++) {
    const particle = document.createElement('div')
    particle.className = 'particle'
    particle.style.left = `${Math.random() * 100}%`
    particle.style.animationDuration = `${12 + Math.random() * 18}s`
    particle.style.animationDelay = `${Math.random() * 15}s`
    particle.style.opacity = `${0.15 + Math.random() * 0.35}`
    canvas.appendChild(particle)
  }
}

// ── Init ──
function init(): void {
  window.addEventListener('DOMContentLoaded', () => {
    spawnParticles()
    setupTheme()
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
//  RUN PIPELINE  (Streaming)
// ══════════════════════════════════════
async function runPipeline(): Promise<void> {
  if (!state.filePath || state.isProcessing) return

  state.isProcessing = true
  window.speechSynthesis.cancel()

  // Reset TTS state for new pipeline run
  state.ttsUtterances = []
  state.ttsCurrentIndex = 0
  state.ttsIsPaused = false
  state.ttsIsSpeaking = false

  const startBtn = document.getElementById('startBtn')!
  const voiceBtn = document.getElementById('voiceBtn')!
  startBtn.setAttribute('disabled', 'true')
  voiceBtn.setAttribute('disabled', 'true')
  showProgress(true)

  // Prepare results panel for incremental rendering
  const phaseResults = document.getElementById('phaseResults')!
  const resultsPanel = document.getElementById('resultsPanel')!
  phaseResults.classList.remove('hidden')
  resultsPanel.innerHTML = ''

  // Reset document image viewer
  const placeholder = document.getElementById('docImagePlaceholder')!
  const viewer = document.getElementById('docImageViewer') as HTMLImageElement
  placeholder.classList.remove('hidden')
  viewer.classList.add('hidden')
  viewer.src = ''

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
  setStatus('Starting pipeline: Upload → Layout → Sort → OCR (per page, streamed)', 'info')

  try {
    // ── Set up SSE event listener BEFORE starting pipeline ──
    // Clean up any previous listener
    if (state.pipelineEventCleanup) {
      state.pipelineEventCleanup()
      state.pipelineEventCleanup = null
    }

    // Register listener for pipeline events from main process
    const cleanup = window.electron.ipcRenderer.on('pipeline-event', (_event, pipelineEvent: PipelineEvent) => {
      handlePipelineEvent(pipelineEvent, timerInterval, progressFill, startTime)
    })
    state.pipelineEventCleanup = cleanup

    // ── Start the pipeline (returns immediately with file_id) ──
    const startResult: StartPipelineResult = await window.electron.ipcRenderer.invoke(
      'start-pipeline',
      state.filePath,
      state.startPage
    )

    console.log('Pipeline started:', startResult)
    setStatus(
      `Pipeline started — ${startResult.total_pages} page(s). Audio will begin as pages are ready…`,
      'info'
    )

  } catch (err) {
    clearInterval(timerInterval)
    progressFill.classList.remove('progress-indeterminate')

    const message = err instanceof Error ? err.message : String(err)
    setStatus(`✗ Pipeline failed: ${message}`, 'error')
    setProgress(0, 'Failed')
    console.error('Pipeline start error:', err)

    finishPipeline()
  }
}


// ══════════════════════════════════════
//  HANDLE STREAMING PIPELINE EVENTS
// ══════════════════════════════════════
function handlePipelineEvent(
  pipelineEvent: PipelineEvent,
  timerInterval: ReturnType<typeof setInterval>,
  progressFill: HTMLElement,
  startTime: number
): void {
  switch (pipelineEvent.type) {
    case 'page_ready': {
      const data = pipelineEvent.data as PageReadyEvent
      console.log(`Page ${data.page}/${data.total_pages} ready (${data.page_time_ms}ms)`)

      // Update progress bar
      const percent = Math.round((data.pages_processed / data.total_pages) * 100)
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0)
      progressFill.classList.remove('progress-indeterminate')
      setProgress(
        percent,
        `📄 Page ${data.page}/${data.total_pages} ready (${elapsed}s elapsed)`
      )

      // Render this page incrementally
      renderPage(data.page_data)

      // Instantiate the image preview immediately if it hasn't been set yet (first page hit)
      const viewer = document.getElementById('docImageViewer') as HTMLImageElement
      if (viewer.classList.contains('hidden')) {
          const imageUrl = data.page_data.debug_image_url ? `http://localhost:8000${data.page_data.debug_image_url}` : ''
          setCurrentDocumentImage(imageUrl)
      }

      // Append to TTS queue and start speaking if not already
      appendPageToTTSQueue(data.page_data)
      if (!state.ttsIsSpeaking) {
        startTTS()
      }

      setStatus(
        `Page ${data.page}/${data.total_pages} ready — ${state.ttsIsSpeaking ? 'reading aloud…' : 'starting audio…'}`,
        'info'
      )
      break
    }

    case 'complete': {
      const data = pipelineEvent.data as PipelineCompleteEvent
      clearInterval(timerInterval)
      progressFill.classList.remove('progress-indeterminate')

      setProgress(100, `✓ Done in ${(data.total_time_ms / 1000).toFixed(1)}s`)
      setStatus(
        `✓ All ${data.total_pages} page(s) processed — ${state.ttsIsSpeaking ? 'reading aloud…' : 'audio complete.'}`,
        'success'
      )

      console.log('Pipeline complete:', data)
      finishPipeline()
      break
    }

    case 'error': {
      const data = pipelineEvent.data as PipelineErrorEvent
      clearInterval(timerInterval)
      progressFill.classList.remove('progress-indeterminate')

      setStatus(`✗ Pipeline error: ${data.error}`, 'error')
      setProgress(0, 'Failed')
      console.error('Pipeline error event:', data)
      finishPipeline()
      break
    }
  }
}

function finishPipeline(): void {
  state.isProcessing = false
  const startBtn = document.getElementById('startBtn')!
  const voiceBtn = document.getElementById('voiceBtn')!
  startBtn.removeAttribute('disabled')
  voiceBtn.removeAttribute('disabled')

  // Clean up event listener
  if (state.pipelineEventCleanup) {
    state.pipelineEventCleanup()
    state.pipelineEventCleanup = null
  }
}


// ══════════════════════════════════════
//  RENDER SINGLE PAGE (Incremental)
// ══════════════════════════════════════
function renderPage(pageData: PipelinePage): void {
  const resultsPanel = document.getElementById('resultsPanel')!

  const pageDiv = document.createElement('div')
  pageDiv.className = 'page-block'

  const title = document.createElement('div')
  title.className = 'page-title'
  title.textContent = `Page ${pageData.page}`
  pageDiv.appendChild(title)

  const sorted = [...pageData.regions].sort(
    (a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)
  )

  for (const region of sorted) {
    const text = region.text || region.gemini_response || ''
    if (!text.trim()) continue

    const label = (region.label || '').toLowerCase().replace(' ', '_')
    const isGemini = GEMINI_LABELS.has(label)

    const block = document.createElement('div')
    block.className = 'region-block'
    block.id = `region-${pageData.page}-${region.reading_order ?? 0}`

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


// ══════════════════════════════════════
//  TEXT-TO-SPEECH  (TTS) — Incremental
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

/**
 * Append a single page's utterances to the TTS queue.
 * Called each time a page_ready event arrives from the backend.
 * If TTS is already playing, the new utterances will be naturally
 * picked up by speakNext() when it reaches them.
 */
function appendPageToTTSQueue(pageData: PipelinePage): void {
  // Page announcement
  const pageAnnounce = new SpeechSynthesisUtterance(`Page ${pageData.page}.`)
  pageAnnounce.rate = 1.0
  state.ttsUtterances.push(pageAnnounce)

  const sorted = [...pageData.regions].sort(
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
      const regionId = `region-${pageData.page}-${region.reading_order ?? 0}`
      const imageUrl = pageData.debug_image_url ? `http://localhost:8000${pageData.debug_image_url}` : ''
      
      utterance.addEventListener('start', () => {
        highlightRegion(regionId, true)
        setCurrentDocumentImage(imageUrl)
      })
      utterance.addEventListener('end', () => highlightRegion(regionId, false))

      state.ttsUtterances.push(utterance)
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
    // No more utterances available right now.
    // If pipeline is still processing, wait for more pages;
    // otherwise, we're truly done.
    if (state.isProcessing) {
      // Pipeline still running — check again shortly for new utterances
      setTTSStatus(`Waiting for next page…`)
      setTimeout(() => {
        if (state.ttsIsSpeaking && !state.ttsIsPaused) {
          speakNext()
        }
      }, 500)
      return
    }

    // Pipeline finished and all utterances spoken
    state.ttsIsSpeaking = false
    if (state.ttsKeepAliveTimer) {
      window.clearInterval(state.ttsKeepAliveTimer)
      state.ttsKeepAliveTimer = null
    }
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

function setCurrentDocumentImage(url: string): void {
  const viewer = document.getElementById('docImageViewer') as HTMLImageElement
  const placeholder = document.getElementById('docImagePlaceholder')!
  
  if (!url) {
    viewer.classList.add('hidden')
    placeholder.classList.remove('hidden')
    return
  }
  
  if (viewer.src !== url) {
    viewer.src = url
    viewer.classList.remove('hidden')
    placeholder.classList.add('hidden')
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
