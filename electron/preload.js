/**
 * DJ Sample Discovery - Electron Preload Script
 * Exposes safe APIs to the renderer process
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Dialog APIs
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  saveFile: (options) => ipcRenderer.invoke('save-file', options),
  openFolder: (path) => ipcRenderer.invoke('open-folder', path),
  
  // File operations
  copyFile: (source, destination) => ipcRenderer.invoke('copy-file', { source, destination }),
  
  // App info
  getAppPath: () => ipcRenderer.invoke('get-app-path'),
  
  // Platform detection
  platform: process.platform,
  
  // Check if running in Electron
  isElectron: true
});
