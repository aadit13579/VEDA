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
  page_queue: number[]
}

// ── Voice parse types ──
interface VoiceSegment {
  from_page: number
  to_page: number        // -1 = 'end'
  from_section: number | null
  to_section:   number | null
}
interface VoiceParseResponse {
  valid: boolean
  reject_reason?: string
  segments?: VoiceSegment[]
  pipeline_segments?: string          // e.g. '5-7,1-3'
  section_filter?: Record<string, [number, number]>  // pageNum → [from, to]
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
  fileId: string | null
  fileCategory: string | null
  startPage: number
  isProcessing: boolean
  isListening: boolean
  // Voice navigation
  pipelineSegments: string | null                          // e.g. '5-7,1-3'
  sectionFilter: Record<string, [number, number]> | null   // pageNum→[fromSec,toSec]
  // Per-region TTS
  activeRegionId: string | null
  regionUtterances: Map<string, SpeechSynthesisUtterance[]>
  regionChunkIndex: Map<string, number>
  ttsKeepAliveTimer: number | null
  // Audio recording (MediaRecorder)
  mediaRecorder: MediaRecorder | null
  audioChunks: Blob[]
  pipelineEventCleanup: (() => void) | null
}

const state: VedaState = {
  filePath: null,
  fileName: null,
  fileId: null,
  fileCategory: null,
  startPage: 1,
  isProcessing: false,
  isListening: false,
  pipelineSegments: null,
  sectionFilter: null,
  activeRegionId: null,
  regionUtterances: new Map(),
  regionChunkIndex: new Map(),
  ttsKeepAliveTimer: null,
  mediaRecorder: null,
  audioChunks: [],
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
function updateThemeIcon(): void {
  const icon = document.getElementById('themeIcon')
  if (!icon) return
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
  icon.textContent = isDark ? '☾' : '☀'
}

function setupTheme(): void {
  const toggleBtn = document.getElementById('themeToggleBtn')!

  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
  const savedTheme = localStorage.getItem('veda-theme')
  const isDark = savedTheme === 'dark' || (!savedTheme && prefersDark)

  if (isDark) document.documentElement.setAttribute('data-theme', 'dark')
  updateThemeIcon()

  toggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme')
    if (currentTheme === 'dark') {
      document.documentElement.removeAttribute('data-theme')
      localStorage.setItem('veda-theme', 'light')
    } else {
      document.documentElement.setAttribute('data-theme', 'dark')
      localStorage.setItem('veda-theme', 'dark')
    }
    updateThemeIcon()
  })
}

// ── Font Size Controls ──
function setupFontSize(): void {
  const sizes = ['sm', 'md', 'lg'] as const
  type FontSize = typeof sizes[number]

  // btnMap must be declared BEFORE applyFontSize is called
  // (applyFontSize reads btnMap — calling it before this line causes a TDZ crash)
  const btnMap: Record<FontSize, string> = {
    sm: 'fontSmBtn',
    md: 'fontMdBtn',
    lg: 'fontLgBtn'
  }

  function applyFontSize(size: FontSize): void {
    document.documentElement.removeAttribute('data-font')
    if (size !== 'md') document.documentElement.setAttribute('data-font', size)
    for (const s of sizes) {
      document.getElementById(btnMap[s])?.classList.toggle('active', s === size)
    }
  }

  // Apply saved preference (safe now that btnMap is defined above)
  const savedFont = (localStorage.getItem('veda-font') ?? 'md') as FontSize
  applyFontSize(savedFont)

  for (const size of sizes) {
    document.getElementById(btnMap[size])?.addEventListener('click', () => {
      applyFontSize(size)
      localStorage.setItem('veda-font', size)
    })
  }
}

// ── Init ──
function init(): void {
  window.addEventListener('DOMContentLoaded', () => {
    setupFontSize()
    setupTheme()
    setupUpload()
    setupVoice()
    setupStart()
    setupTTS()
    setupHistory()
    setupNewDocBtn()
    checkBackendHealth()
  })
}

// ── Reset to Upload ──
function setupNewDocBtn(): void {
  const newDocBtn = document.getElementById('newDocBtn')
  if (!newDocBtn) return

  newDocBtn.addEventListener('click', () => {
    // Cancel any ongoing audio/processing
    window.speechSynthesis.cancel()
    if (state.pipelineEventCleanup) {
      state.pipelineEventCleanup()
      state.pipelineEventCleanup = null
    }

    // Reset state
    state.fileId = null
    state.filePath = null
    state.fileName = ''
    state.activeRegionId = null
    state.regionUtterances.clear()
    state.regionChunkIndex.clear()
    state.isProcessing = false

    // Reset UI visibility
    document.getElementById('phaseUpload')?.classList.remove('hidden')
    document.getElementById('phaseResults')?.classList.add('hidden')
    document.getElementById('newDocBtn')?.classList.add('hidden')
    document.getElementById('progressContainer')?.classList.add('hidden')

    // Clear upload state
    document.getElementById('fileBadge')?.classList.add('hidden')
    document.getElementById('optionsRow')?.classList.add('hidden')
    document.getElementById('voiceFeedback')?.classList.add('hidden')

    // Clear results
    const resultsPanel = document.getElementById('resultsPanel')
    if (resultsPanel) resultsPanel.innerHTML = ''

    // Rebuild image viewer container to clear past images
    const docImageContainer = document.getElementById('docImageContainer')
    if (docImageContainer) {
      docImageContainer.innerHTML = `
        <p id="docImagePlaceholder">Processing your document…</p>
        <iframe id="docPdfViewer" class="hidden" title="Document Preview" aria-label="PDF document viewer"></iframe>
        <img id="docImageViewer" class="hidden" src="" alt="Document page" />
      `
    }

    setStatus('Ready to upload a document.', 'info')
  })
}

// ── Backend Health Check ──
async function checkBackendHealth(): Promise<void> {
  try {
    const isHealthy = await window.electron.ipcRenderer.invoke('check-backend')
    if (isHealthy) {
      setStatus('Backend connected — ready to upload a document.', 'success')
    } else {
      setStatus('Backend is not running. Start it with: uvicorn main:app --reload', 'warning')
    }
  } catch {
    setStatus('Backend is not running. Start it with: uvicorn main:app --reload', 'warning')
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

        setStatus(`"${state.fileName}" selected — ready to start.`, 'info')
      }
    } catch (err) {
      console.error('File dialog error:', err)
      setStatus('Error opening file dialog.', 'error')
    }
  })
}

// ══════════════════════════════════════
//  VOICE COMMAND  (MediaRecorder → WAV → backend /transcribe)
// ══════════════════════════════════════
function setupVoice(): void {
  const voiceBtn = document.getElementById('voiceBtn')!

  voiceBtn.addEventListener('click', async () => {
    if (state.isProcessing) return

    // If already recording — stop
    if (state.isListening) {
      if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        state.mediaRecorder.stop()
      }
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      // Pick best supported MIME type
      let mimeType = 'audio/webm;codecs=opus'
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm'
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = ''
      }

      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      state.mediaRecorder = mediaRecorder
      state.audioChunks   = []
      state.isListening   = true
      updateVoiceUI(true)

      mediaRecorder.ondataavailable = (e: BlobEvent) => {
        if (e.data.size > 0) state.audioChunks.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop())   // release mic
        state.isListening = false
        updateVoiceUI(false)

        const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' })
        if (audioBlob.size === 0) {
          setStatus('No audio recorded. Try again.', 'warning')
          return
        }

        setStatus('Transcribing…', 'info')
        try {
          // WebM → 16 kHz mono WAV  (backend expects WAV)
          const wavBuffer = await convertBlobToWav(audioBlob)

          const transcript: string = await window.electron.ipcRenderer.invoke(
            'transcribe-audio',
            wavBuffer
          )

          console.log('Transcript:', transcript, '(length:', transcript.length, ')')

          if (!transcript.trim()) {
            setStatus('No speech detected. Try again.', 'warning')
            return
          }

          showVoiceTranscript(transcript)
          setStatus('Parsing voice command…', 'info')

          const parsed = await parseVoiceCommand(transcript)
          if (!parsed.valid) {
            showVoiceReject(parsed.reject_reason ?? 'Invalid command.')
            setStatus(`Voice rejected: ${parsed.reject_reason}`, 'warning')
          } else {
            state.pipelineSegments = parsed.pipeline_segments ?? null
            state.sectionFilter    = parsed.section_filter   ?? null
            state.startPage        = parsed.segments?.[0]?.from_page ?? 1
            showVoiceQueue(parsed)
            setStatus(`Voice parsed — pages: ${parsed.pipeline_segments}. Starting…`, 'info')
            runPipeline()
          }
        } catch (err) {
          console.error('Voice pipeline error:', err)
          setStatus('Voice command failed. Is the backend running?', 'error')
        }
      }

      mediaRecorder.start()
      setStatus('Recording… speak your command, then click the button again to stop.', 'info')

      // Auto-stop after 8 s so the user doesn’t have to remember to click
      setTimeout(() => {
        if (state.mediaRecorder?.state === 'recording') state.mediaRecorder.stop()
      }, 8000)

    } catch (err) {
      console.error('Mic error:', err)
      setStatus('Microphone access denied. Check your permissions.', 'error')
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
    voiceBtn.dataset.recording = 'true'
    voicePulse.classList.remove('hidden')
    voiceLabel.textContent = 'Recording… click to stop'
  } else {
    delete voiceBtn.dataset.recording
    voicePulse.classList.add('hidden')
    voiceLabel.textContent = 'Go to a specific page by voice'
  }
}

// ══════════════════════════════════════
//  VOICE COMMAND PARSER  (calls backend)
// ══════════════════════════════════════

/** POST /voice/parse — returns structured command or rejection. */
async function parseVoiceCommand(transcript: string): Promise<VoiceParseResponse> {
  const res = await fetch('http://localhost:8000/api/v1/voice/parse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript }),
  })
  if (!res.ok) throw new Error(`/voice/parse returned ${res.status}`)
  return res.json() as Promise<VoiceParseResponse>
}

/** Show the raw transcript in the feedback panel. */
function showVoiceTranscript(transcript: string): void {
  const fb = document.getElementById('voiceFeedback')!
  const el = document.getElementById('voiceTranscript')!
  fb.classList.remove('hidden')
  el.textContent = `"${transcript}"`
  document.getElementById('parsedPage')!.textContent = ''
  document.getElementById('voiceQueueSummary')!.textContent = ''
  document.getElementById('voiceRejectBanner')!.classList.add('hidden')
  document.getElementById('voiceRejectBanner')!.classList.remove('voice-reject-shake')
}

/** Show the parsed queue in the feedback panel. */
function showVoiceQueue(parsed: VoiceParseResponse): void {
  const summary = document.getElementById('voiceQueueSummary')!
  const parsedPage = document.getElementById('parsedPage')!

  // Build readable page order string
  const pages: string[] = []
  for (const seg of parsed.segments ?? []) {
    const to = seg.to_page === -1 ? 'end' : String(seg.to_page)
    const range = seg.from_page === seg.to_page ? String(seg.from_page) : `${seg.from_page}–${to}`
    const secPart = seg.from_section != null
      ? ` (sec ${seg.from_section}${seg.to_section !== seg.from_section ? '–' + seg.to_section : ''})`
      : ''
    pages.push(`Page ${range}${secPart}`)
  }

  summary.textContent = pages.join(' → ')
  parsedPage.textContent = `Queue: ${parsed.pipeline_segments}`
}

/** Show rejection banner with shake animation. */
function showVoiceReject(reason: string): void {
  const banner = document.getElementById('voiceRejectBanner')!
  banner.textContent = `⚠ ${reason}`
  banner.classList.remove('hidden')
  // Trigger shake — remove then re-add class
  banner.classList.remove('voice-reject-shake')
  requestAnimationFrame(() => {
    requestAnimationFrame(() => banner.classList.add('voice-reject-shake'))
  })
  document.getElementById('parsedPage')!.textContent = ''
  document.getElementById('voiceQueueSummary')!.textContent = ''
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

    // Full document — clear any voice-set segments/filters
    state.startPage        = 1
    state.pipelineSegments = null
    state.sectionFilter    = null
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

  state.activeRegionId = null
  state.regionUtterances.clear()
  state.regionChunkIndex.clear()

  const startBtn = document.getElementById('startBtn')!
  const voiceBtn = document.getElementById('voiceBtn')!
  startBtn.setAttribute('disabled', 'true')
  voiceBtn.setAttribute('disabled', 'true')
  showProgress(true)

  const phaseResults = document.getElementById('phaseResults')!
  const resultsPanel = document.getElementById('resultsPanel')!
  phaseResults.classList.remove('hidden')
  resultsPanel.innerHTML = ''

  setDocumentViewer(null, null)

  const progressFill = document.getElementById('progressFill')!
  progressFill.classList.add('progress-indeterminate')
  const startTime = Date.now()

  // Label for progress — show segments string if available, else start page
  const modeLabel = state.pipelineSegments
    ? `pages ${state.pipelineSegments}`
    : `from page ${state.startPage}`

  const timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0)
    setProgressText(`Processing "${state.fileName}" (${modeLabel})… (${elapsed}s)`)
  }, 1000)
  setProgressText(`Processing "${state.fileName}" (${modeLabel})…`)
  setStatus('Analysing document — pages will appear as they are ready.', 'info')

  try {
    if (state.pipelineEventCleanup) {
      state.pipelineEventCleanup()
      state.pipelineEventCleanup = null
    }

    const cleanup = window.electron.ipcRenderer.on('pipeline-event', (_event, pipelineEvent: PipelineEvent) => {
      handlePipelineEvent(pipelineEvent, timerInterval, progressFill, startTime)
    })
    state.pipelineEventCleanup = cleanup

    // Build IPC args — pass segments string when available
    const startResult: StartPipelineResult = await window.electron.ipcRenderer.invoke(
      'start-pipeline',
      state.filePath,
      state.startPage,
      state.pipelineSegments   // new optional 3rd arg — undefined when not set
    )

    state.fileId       = startResult.file_id
    state.fileCategory = startResult.category
    setDocumentViewer(startResult.file_id, startResult.category)

    console.log('Pipeline started:', startResult)
    setStatus(
      `Processing — ${startResult.total_pages} page(s) detected. Audio will begin shortly…`,
      'info'
    )

  } catch (err) {
    clearInterval(timerInterval)
    progressFill.classList.remove('progress-indeterminate')
    const message = err instanceof Error ? err.message : String(err)
    setStatus(`Processing failed: ${message}`, 'error')
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
        `Page ${data.page} of ${data.total_pages} complete (${elapsed}s)`
      )

      // Render this page (creates region blocks with per-region play buttons)
      renderPage(data.page_data)

      setStatus(
        `Page ${data.page} of ${data.total_pages} processed.`,
        'info'
      )
      break
    }

    case 'complete': {
      const data = pipelineEvent.data as PipelineCompleteEvent
      clearInterval(timerInterval)
      progressFill.classList.remove('progress-indeterminate')

      setProgress(100, `Done in ${(data.total_time_ms / 1000).toFixed(1)}s`)
      setStatus(
        `All ${data.total_pages} page(s) processed successfully.`,
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

      setStatus(`An error occurred: ${data.error}`, 'error')
      setProgress(0, 'Processing failed')
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

  // Section filter: [fromSec, toSec] (1-indexed reading_order) or null
  const sf = state.sectionFilter?.[String(pageData.page)] ?? null

  const pageDiv = document.createElement('div')
  pageDiv.className = 'page-block'

  const title = document.createElement('div')
  title.className = 'page-title'
  title.textContent = sf
    ? `Page ${pageData.page}  (sections ${sf[0]}–${sf[1]})`
    : `Page ${pageData.page}`
  pageDiv.appendChild(title)

  const sorted = [...pageData.regions].sort(
    (a, b) => (a.reading_order ?? 0) - (b.reading_order ?? 0)
  )

  // reading_order is 0-indexed in data; compare as 1-indexed for the filter
  for (const region of sorted) {
    const regionOrder1 = (region.reading_order ?? 0) + 1
    if (sf && (regionOrder1 < sf[0] || regionOrder1 > sf[1])) continue
    const text = region.text || region.gemini_response || ''
    if (!text.trim()) continue

    const label = (region.label || '').toLowerCase().replace(' ', '_')
    const isGemini = GEMINI_LABELS.has(label)
    const regionId = `region-${pageData.page}-${region.reading_order ?? 0}`

    const block = document.createElement('div')
    block.className = 'region-block'
    block.id = regionId

    // ── Label row with inline Play/Pause button ──
    const labelRow = document.createElement('div')
    labelRow.className = 'region-label-row'

    const labelEl = document.createElement('span')
    labelEl.className = `region-label ${isGemini ? 'gemini' : ''}`
    labelEl.textContent = isGemini ? `${region.label} (visual)` : region.label || 'text'
    labelRow.appendChild(labelEl)

    // Play/Pause toggle button for this region
    const playBtn = document.createElement('button')
    playBtn.className = 'region-play-btn'
    playBtn.setAttribute('aria-label', 'Play this section')
    playBtn.dataset.regionId = regionId
    playBtn.innerHTML = `
      <svg class="icon-play" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      <svg class="icon-pause hidden" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="14" height="14"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
      <span class="region-play-label">Play</span>
    `
    playBtn.addEventListener('click', () => toggleRegionPlayback(regionId))
    labelRow.appendChild(playBtn)
    block.appendChild(labelRow)

    const textEl = document.createElement('div')
    textEl.className = 'region-text'
    textEl.textContent = text.trim()
    block.appendChild(textEl)

    pageDiv.appendChild(block)

    // Pre-build utterances for this region and store in state
    buildRegionUtterances(regionId, text.trim(), isGemini ? region.label || '' : '')
  }

  resultsPanel.appendChild(pageDiv)
}



// ══════════════════════════════════════
//  PER-REGION TTS
// ══════════════════════════════════════

function setupTTS(): void {
  // No global buttons — per-region buttons are wired in renderPage()
}

/** Build and store SpeechSynthesisUtterance chunks for a region. */
function buildRegionUtterances(regionId: string, text: string, labelPrefix: string): void {
  let speakText = text.trim()
  if (labelPrefix) speakText = `${labelPrefix} description: ${speakText}`
  const chunks = splitTextForTTS(speakText, 30).map(chunk => {
    const u = new SpeechSynthesisUtterance(chunk)
    u.rate = 1.0
    u.lang = 'en-US'
    return u
  })
  state.regionUtterances.set(regionId, chunks)
  state.regionChunkIndex.set(regionId, 0)
}

/** Play/Pause toggle for a region's Play button. */
function toggleRegionPlayback(regionId: string): void {
  if (state.activeRegionId === regionId) {
    // This region is already active — toggle pause/resume
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume()
      setRegionButtonState(regionId, 'playing')
      setTTSStatus('Reading…')
    } else {
      window.speechSynthesis.pause()
      setRegionButtonState(regionId, 'paused')
      setTTSStatus('Paused')
    }
    return
  }

  // Stop whatever else is playing and start this region from scratch
  stopActiveRegion()

  const utterances = state.regionUtterances.get(regionId)
  if (!utterances || utterances.length === 0) return

  state.activeRegionId = regionId
  state.regionChunkIndex.set(regionId, 0)
  setRegionButtonState(regionId, 'playing')
  highlightRegion(regionId, true)

  // Keep-alive to prevent Chrome's 15-second silence cutoff
  if (state.ttsKeepAliveTimer) window.clearInterval(state.ttsKeepAliveTimer)
  state.ttsKeepAliveTimer = window.setInterval(() => {
    if (state.activeRegionId && !window.speechSynthesis.paused) {
      window.speechSynthesis.pause()
      window.speechSynthesis.resume()
    }
  }, 10000)

  speakRegionChunk(regionId)
}

/** Speak one chunk; auto-advances through all chunks for the region. */
function speakRegionChunk(regionId: string): void {
  if (state.activeRegionId !== regionId) return

  const utterances = state.regionUtterances.get(regionId)!
  const idx = state.regionChunkIndex.get(regionId) ?? 0

  if (idx >= utterances.length) {
    stopActiveRegion()
    setTTSStatus('Done')
    return
  }

  const utterance = utterances[idx]
  utterance.onend = () => {
    if (state.activeRegionId !== regionId) return
    state.regionChunkIndex.set(regionId, idx + 1)
    speakRegionChunk(regionId)
  }
  utterance.onerror = (e) => {
    console.error('TTS chunk error:', e)
    if (state.activeRegionId !== regionId) return
    state.regionChunkIndex.set(regionId, idx + 1)
    speakRegionChunk(regionId)
  }

  const total = utterances.length
  setTTSStatus(total > 1 ? `Reading (${idx + 1}/${total})…` : 'Reading…')
  window.speechSynthesis.speak(utterance)
}

/** Cancel the active region's playback and reset its button. */
function stopActiveRegion(): void {
  window.speechSynthesis.cancel()
  if (state.ttsKeepAliveTimer) {
    window.clearInterval(state.ttsKeepAliveTimer)
    state.ttsKeepAliveTimer = null
  }
  if (state.activeRegionId) {
    setRegionButtonState(state.activeRegionId, 'stopped')
    highlightRegion(state.activeRegionId, false)
  }
  state.activeRegionId = null
}

/** Swap the Play/Pause icon and label on a region's button. */
function setRegionButtonState(regionId: string, mode: 'playing' | 'paused' | 'stopped'): void {
  const block = document.getElementById(regionId)
  if (!block) return
  const btn     = block.querySelector<HTMLElement>('.region-play-btn')
  const iconPl  = btn?.querySelector('.icon-play')
  const iconPa  = btn?.querySelector('.icon-pause')
  const lbl     = btn?.querySelector<HTMLElement>('.region-play-label')
  if (!btn) return

  if (mode === 'playing') {
    iconPl?.classList.add('hidden')
    iconPa?.classList.remove('hidden')
    if (lbl) lbl.textContent = 'Pause'
    btn.setAttribute('aria-label', 'Pause this section')
  } else {
    iconPl?.classList.remove('hidden')
    iconPa?.classList.add('hidden')
    if (lbl) lbl.textContent = mode === 'paused' ? 'Resume' : 'Play'
    btn.setAttribute('aria-label', mode === 'paused' ? 'Resume this section' : 'Play this section')
  }
}

function highlightRegion(regionId: string, active: boolean): void {
  const el = document.getElementById(regionId)
  if (!el) return
  if (active) {
    document.querySelectorAll('.region-block.speaking').forEach(other => other.classList.remove('speaking'))
    el.classList.add('speaking')
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  } else {
    el.classList.remove('speaking')
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

// ══════════════════════════════════════
//  DOCUMENT VIEWER
// ══════════════════════════════════════

/**
 * Show the original uploaded document in the right pane.
 * PDFs use an <iframe> for native browser rendering.
 * Images use an <img>.
 * Pass null/null to reset to the placeholder.
 */
function setDocumentViewer(fileId: string | null, category: string | null): void {
  const placeholder = document.getElementById('docImagePlaceholder')!
  const pdfViewer   = document.getElementById('docPdfViewer') as HTMLIFrameElement
  const imgViewer   = document.getElementById('docImageViewer') as HTMLImageElement

  if (!fileId) {
    placeholder.classList.remove('hidden')
    pdfViewer.classList.add('hidden')
    imgViewer.classList.add('hidden')
    pdfViewer.src = ''
    imgViewer.src = ''
    return
  }

  const docUrl = `http://localhost:8000/api/v1/document/${fileId}`
  const isImage = category?.startsWith('IMAGE') ?? false

  placeholder.classList.add('hidden')

  if (isImage) {
    pdfViewer.classList.add('hidden')
    pdfViewer.src = ''
    imgViewer.src = docUrl
    imgViewer.classList.remove('hidden')
  } else {
    imgViewer.classList.add('hidden')
    imgViewer.src = ''
    // #toolbar=0&navpanes=0 hides Chrome's built-in PDF toolbar clutter
    pdfViewer.src = docUrl + '#toolbar=0&navpanes=0'
    pdfViewer.classList.remove('hidden')
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
  const bar = document.getElementById('statusBar')
  const el  = document.getElementById('statusText')
  if (!el) return

  el.textContent = msg
  el.className = '' // clear previous colour classes

  switch (type) {
    case 'success': el.classList.add('status-success'); break
    case 'warning': el.classList.add('status-warning'); break
    case 'error':   el.classList.add('status-error');   break
    default:        el.classList.add('status-info');     break
  }

  if (bar) bar.classList.remove('hidden')
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

// ══════════════════════════════════════
//  HISTORY PANEL
// ══════════════════════════════════════

interface HistoryItem {
  file_id: string
  filename: string
  category: string
  total_pages: number
  timestamp: number
}

function setupHistory(): void {
  const toggleBtn = document.getElementById('historyToggleBtn')
  const closeBtn = document.getElementById('closeHistoryBtn')
  const panel = document.getElementById('historyPanel')

  if (!toggleBtn || !closeBtn || !panel) return

  toggleBtn.addEventListener('click', () => {
    const isOpen = panel.classList.contains('open')
    if (isOpen) {
      panel.classList.remove('open')
    } else {
      panel.classList.add('open')
      loadHistoryItems()
    }
  })

  closeBtn.addEventListener('click', () => {
    panel.classList.remove('open')
  })
}

async function loadHistoryItems(): Promise<void> {
  const list = document.getElementById('historyList')
  if (!list) return

  try {
    const res = await fetch('http://localhost:8000/api/v1/pipeline/history')
    if (!res.ok) throw new Error('Failed to fetch history')
    const items: HistoryItem[] = await res.json()

    list.innerHTML = ''
    if (items.length === 0) {
      list.innerHTML = '<p class="history-empty">No previous documents found.</p>'
      return
    }

    items.forEach(item => {
      const el = document.createElement('div')
      el.className = 'history-item'
      
      const title = document.createElement('div')
      title.className = 'history-item-title'
      title.textContent = item.filename
      
      const meta = document.createElement('div')
      meta.className = 'history-item-meta'
      const date = new Date(item.timestamp * 1000).toLocaleString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
      })
      meta.textContent = `${item.total_pages} page(s) • ${date}`
      
      el.appendChild(title)
      el.appendChild(meta)
      
      el.addEventListener('click', () => loadHistoryDocument(item))
      list.appendChild(el)
    })
  } catch (err) {
    console.error('History API error:', err)
    list.innerHTML = '<p class="history-error">Failed to load history.</p>'
  }
}

async function loadHistoryDocument(item: HistoryItem): Promise<void> {
  document.getElementById('historyPanel')?.classList.remove('open')
  
  if (state.isProcessing) {
    setStatus('Cannot load history while processing another document.', 'warning')
    return
  }
  
  setStatus(`Loading "${item.filename}" from history…`, 'info')
  
  try {
    const res = await fetch(`http://localhost:8000/api/v1/pipeline/history/${item.file_id}`)
    if (!res.ok) throw new Error('Failed to fetch document JSON')
    
    const docData = await res.json()
    
    // Clear state
    state.filePath = null // It's not a local path anymore
    state.fileName = item.filename
    state.fileId = item.file_id
    state.fileCategory = item.category
    
    window.speechSynthesis.cancel()
    state.activeRegionId = null
    state.regionUtterances.clear()
    state.regionChunkIndex.clear()
    
    // Prepare UI
    const phaseUpload = document.getElementById('phaseUpload')!
    const phaseResults = document.getElementById('phaseResults')!
    const resultsPanel = document.getElementById('resultsPanel')!
    
    phaseUpload.classList.add('hidden')
    phaseResults.classList.remove('hidden')
    document.getElementById('newDocBtn')?.classList.remove('hidden')
    resultsPanel.innerHTML = ''
    
    // Show document in viewer
    setDocumentViewer(item.file_id, item.category)
    
    // Render all pages incrementally
    const pages: PipelinePage[] = docData.pages || []
    pages.forEach(page => renderPage(page))
    
    setStatus(`"${item.filename}" loaded successfully from library.`, 'success')
    setTTSStatus('Ready')
    
  } catch (err) {
    console.error('Failed to load history document:', err)
    setStatus('Failed to load document from history.', 'error')
  }
}

init()
