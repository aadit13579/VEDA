import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
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

  // ── IPC: Open file dialog ──
  ipcMain.handle('open-file-dialog', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select Documents',
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Documents', extensions: ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'tif'] },
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
  ipcMain.handle('start-pipeline', async (_event, filePath: string, startPage: number = 1) => {
    const mainWindow = BrowserWindow.getAllWindows()[0]
    if (!mainWindow) throw new Error('No main window found')

    // ── Step 1: Upload file to /pipeline/start ──
    const fileName = path.basename(filePath)
    const fileBuffer = fs.readFileSync(filePath)

    const formData = new FormData()
    const blob = new Blob([fileBuffer])
    formData.append('file', blob, fileName)

    const startUrl = new URL(`${BACKEND_URL}/api/v1/pipeline/start`)
    startUrl.searchParams.append('start_page', startPage.toString())

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

  // ── IPC: Check backend health ──
  ipcMain.handle('check-backend', async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/`)
      return response.ok
    } catch {
      return false
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
