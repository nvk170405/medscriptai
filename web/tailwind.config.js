/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      colors: {
        background: '#050505',
        surface: '#0a0a0a',
        surfaceHighlight: 'rgba(255, 255, 255, 0.05)',
        glow: '#7da291',
        glowLight: '#9bbbaa',
        muted: '#8a8a8e',
      }
    },
  },
  plugins: [],
}
