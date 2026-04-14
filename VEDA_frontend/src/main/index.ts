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

  // ── IPC: Run Document Pipeline ──
  ipcMain.handle('run-pipeline', async (_event, filePath: string, startPage: number = 1) => {
    try {
      const fileName = path.basename(filePath)
      const fileBuffer = fs.readFileSync(filePath)

      // Build multipart form data manually using fetch (Node 18+)
      const formData = new FormData()
      const blob = new Blob([fileBuffer])
      formData.append('file', blob, fileName)

      const url = new URL(`${BACKEND_URL}/api/v1/pipeline`)
      url.searchParams.append('start_page', startPage.toString())

      const response = await fetch(url.toString(), {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        let errorText = await response.text()
        // Try parsing JSON error for better message
        try {
          const parsed = JSON.parse(errorText)
          if (parsed.detail) {
            errorText = typeof parsed.detail === 'string' ? parsed.detail : JSON.stringify(parsed.detail)
          }
        } catch { }
        throw new Error(`Pipeline failed (${response.status}): ${errorText}`)
      }

      return await response.json()
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error)
      throw new Error(`Pipeline error: ${message}`)
    }
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
