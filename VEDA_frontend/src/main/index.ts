import { app, shell, BrowserWindow, ipcMain, dialog, session } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import * as fs from 'fs'
import * as path from 'path'

const BACKEND_URL = 'http://localhost:8000'

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#0a0e1a',
    icon,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('com.veda')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // ── Auto-grant microphone permission for Web Speech API ──
  session.defaultSession.setPermissionRequestHandler((_webContents, permission, callback) => {
    const allowed = ['media', 'microphone', 'audio-capture']
    callback(allowed.includes(permission))
  })
  session.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    const allowed = ['media', 'microphone', 'audio-capture']
    return allowed.includes(permission)
  })

  // ── IPC: Open file dialog ──
  ipcMain.handle('open-file-dialog', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select Document',
      properties: ['openFile'],
      filters: [
        {
          name: 'Documents',
          extensions: ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'tif', 'doc', 'docx', 'ppt', 'pptx']
        },
        { name: 'All Files', extensions: ['*'] }
      ]
    })
    return result.filePaths
  })

  // ── IPC: Start Pipeline (streaming) ──
  // 1. Uploads file to POST /pipeline/start → gets file_id + total_pages
  // 2. Connects to GET /pipeline/stream/{file_id} (SSE)
  // 3. Forwards each SSE event to the renderer via webContents.send()
  // 4. Returns the initial { file_id, total_pages, ... } to the renderer immediately
  ipcMain.handle('start-pipeline', async (_event, filePath: string, startPage: number = 1, segments?: string | null) => {
    const mainWindow = BrowserWindow.getAllWindows()[0]
    if (!mainWindow) throw new Error('No main window found')

    // ── Step 1: Upload file to /pipeline/start ──
    const fileName = path.basename(filePath)
    const fileBuffer = fs.readFileSync(filePath)

    const formData = new FormData()
    const blob = new Blob([fileBuffer])
    formData.append('file', blob, fileName)

    const startUrl = new URL(`${BACKEND_URL}/api/v1/pipeline/start`)

    if (segments) {
      // Voice-driven non-sequential mode: segments overrides start_page
      startUrl.searchParams.append('segments', segments)
    } else {
      startUrl.searchParams.append('start_page', startPage.toString())
    }

    const startResponse = await fetch(startUrl.toString(), {
      method: 'POST',
      body: formData
    })

    if (!startResponse.ok) {
      let errorText = await startResponse.text()
      try {
        const parsed = JSON.parse(errorText)
        if (parsed.detail) {
          errorText = typeof parsed.detail === 'string' ? parsed.detail : JSON.stringify(parsed.detail)
        }
      } catch { }
      throw new Error(`Pipeline start failed (${startResponse.status}): ${errorText}`)
    }

    const startResult = await startResponse.json()
    const fileId = startResult.file_id

    // ── Step 2: Connect to SSE stream in the background ──
    // This runs async — we don't await it. Events are pushed to renderer.
    connectToSSEStream(fileId, mainWindow).catch((err) => {
      console.error('SSE stream error:', err)
      mainWindow.webContents.send('pipeline-event', {
        type: 'error',
        data: {
          file_id: fileId,
          error: err instanceof Error ? err.message : String(err),
          pages_processed: 0
        }
      })
    })

    // ── Step 3: Return immediately so renderer can set up listeners ──
    return startResult
  })


  // ── IPC: Run full pipeline (single backend call) ──
  ipcMain.handle(
    'run-pipeline',
    async (_event, filePath: string, startPage: number = 1) => {
      try {
        const fileName = path.basename(filePath)
        const fileBuffer = fs.readFileSync(filePath)

        // Determine MIME type from file extension
        const ext = path.extname(fileName).toLowerCase()
        const mimeTypes: Record<string, string> = {
          '.pdf': 'application/pdf',
          '.png': 'image/png',
          '.jpg': 'image/jpeg',
          '.jpeg': 'image/jpeg',
          '.tiff': 'image/tiff',
          '.tif': 'image/tiff',
          '.doc': 'application/msword',
          '.docx':
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          '.ppt': 'application/vnd.ms-powerpoint',
          '.pptx':
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        }
        const mimeType = mimeTypes[ext] || 'application/octet-stream'

        // Build multipart form data with a proper File object (includes filename + MIME)
        const formData = new FormData()
        const file = new File([new Uint8Array(fileBuffer)], fileName, { type: mimeType })
        formData.append('file', file)

        const url = `${BACKEND_URL}/api/v1/pipeline?start_page=${startPage}`

        const response = await fetch(url, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorBody = await response.text()
          throw new Error(`Pipeline failed (${response.status}): ${errorBody}`)
        }

        const pipelineResult = await response.json()
        return pipelineResult
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error)
        throw new Error(`Pipeline error: ${message}`)
      }
    }
  )

  // ── IPC: Check backend health ──
  ipcMain.handle('check-backend', async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/`)
      return response.ok
    } catch {
      return false
    }
  })

  // ── IPC: Transcribe audio (send to backend Gemini endpoint) ──
  ipcMain.handle('transcribe-audio', async (_event, audioData: ArrayBuffer) => {
    try {
      const formData = new FormData()
      const file = new File([new Uint8Array(audioData)], 'recording.webm', {
        type: 'audio/webm'
      })
      formData.append('file', file)

      const response = await fetch(`${BACKEND_URL}/api/v1/transcribe`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorBody = await response.text()
        throw new Error(`Transcription failed (${response.status}): ${errorBody}`)
      }

      const result = (await response.json()) as { transcript?: string }
      return result.transcript || ''
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error)
      throw new Error(`Transcription error: ${message}`)
    }
  })

  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})


// ── SSE Stream Reader ──
// Connects to the backend SSE endpoint and forwards parsed events to the renderer.
async function connectToSSEStream(fileId: string, mainWindow: BrowserWindow): Promise<void> {
  const streamUrl = `${BACKEND_URL}/api/v1/pipeline/stream/${fileId}`
  console.log(`[SSE] Connecting to ${streamUrl}`)

  const response = await fetch(streamUrl)

  if (!response.ok) {
    throw new Error(`SSE connection failed (${response.status})`)
  }

  if (!response.body) {
    throw new Error('SSE response has no body')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      console.log('[SSE] Stream closed by server')
      break
    }

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE frames: each frame is "event: <type>\ndata: <json>\n\n"
    const frames = buffer.split('\n\n')
    // The last element is either empty or an incomplete frame
    buffer = frames.pop() || ''

    for (const frame of frames) {
      if (!frame.trim()) continue

      const lines = frame.split('\n')
      let eventType = ''
      let eventData = ''

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim()
        } else if (line.startsWith('data: ')) {
          eventData = line.slice(6)
        }
      }

      if (eventType && eventData) {
        try {
          const parsed = JSON.parse(eventData)
          console.log(`[SSE] Event: ${eventType}`, eventType === 'page_ready'
            ? `page ${parsed.page}/${parsed.total_pages}`
            : '')

          mainWindow.webContents.send('pipeline-event', {
            type: eventType,
            data: parsed
          })
        } catch (err) {
          console.error('[SSE] Failed to parse event data:', eventData, err)
        }
      }
    }
  }
}
