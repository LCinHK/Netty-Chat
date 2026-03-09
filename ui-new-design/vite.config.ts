import { defineConfig } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Top-level config options
  base: '/ui/',  // ← This makes all asset paths start with /ui/ (critical for FastAPI mounting)

  plugins: [
    // React and Tailwind plugins (required for the project)
    react(),
    tailwindcss(),
  ],

  resolve: {
    alias: {
      // Alias @ to src/ directory (common in React/Vite projects)
      '@': path.resolve(__dirname, './src'),
    },
  },

  // Allow importing these file types as raw strings (e.g. import svg as raw)
  assetsInclude: ['**/*.svg', '**/*.csv'],
})