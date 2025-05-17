import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
    plugins: [vue(), tailwindcss()],
    server: {
        proxy: {
            '/api': {
                target: 'http://127.0.0.1:8000',
                changeOrigin: true,
                secure: false,
            },
            'http://127.0.0.1:8200/intake': {
                target: 'http://localhost:8200',
                changeOrigin: true,
                secure: false,
            },
        },
    },
})
