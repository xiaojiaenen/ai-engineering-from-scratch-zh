const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const PORT = 3300;
const SITE_URL = `http://localhost:${PORT}/site/`;

let serverProcess = null;
let mainWindow = null;

// ── Start Python HTTP server ──────────────────────────────
function startServer() {
  return new Promise((resolve, reject) => {
    serverProcess = spawn('python3', ['-m', 'http.server', String(PORT)], {
      cwd: PROJECT_ROOT,
      stdio: 'ignore',
    });

    serverProcess.on('error', (err) => {
      reject(new Error(`无法启动服务器: ${err.message}`));
    });

    // Wait for server to be ready
    let attempts = 0;
    const check = () => {
      http.get(SITE_URL, (res) => {
        resolve();
      }).on('error', () => {
        attempts++;
        if (attempts > 30) {
          reject(new Error('服务器启动超时'));
        } else {
          setTimeout(check, 300);
        }
      });
    };
    setTimeout(check, 500);
  });
}

// ── Create main window ─────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'AI Engineering from Scratch',
    icon: path.join(PROJECT_ROOT, 'assets', 'icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadURL(SITE_URL);

  // Open external links in system browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('http://localhost')) return { action: 'allow' };
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── App lifecycle ──────────────────────────────────────────
app.whenReady().then(async () => {
  try {
    await startServer();
    console.log(`Server ready at http://localhost:${PORT}`);
    createWindow();
  } catch (err) {
    console.error('启动失败:', err.message);
    // Try to create window anyway (server might already be running)
    createWindow();
  }
});

app.on('window-all-closed', () => {
  if (serverProcess) {
    serverProcess.kill();
  }
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

app.on('before-quit', () => {
  if (serverProcess) {
    serverProcess.kill();
  }
});
