import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
    proxy: {
      '/api': {
        // 固定走 IPv4，避免 localhost 在本机解析不稳定时导致代理报错。
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
})
