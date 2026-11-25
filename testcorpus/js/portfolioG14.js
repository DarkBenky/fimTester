import { fileURLToPath, URL } from 'node:url'

import { defineConfig, searchForWorkspaceRoot } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import vuetify from 'vite-plugin-vuetify'

export default defineConfig(({ mode }) => {
  return {
    plugins: [
      vue(),
      vuetify({ autoImport: true }),
      ...(mode === 'development' ? [vueDevTools()] : []),
    ],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      },
    },
    build: {
      outDir: fileURLToPath(new URL('../internal/webui/assets', import.meta.url)),
      emptyOutDir: true,
    },
    server: {
      host: '127.0.0.1',
      port: 5173,
      strictPort: true,
      fs: {
        strict: true,
        allow: [searchForWorkspaceRoot(process.cwd())],
      },
      allowedHosts: ['localhost', '127.0.0.1'],
    },
  }
})
