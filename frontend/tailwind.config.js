/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: '#ffffff',
          alt: '#f9fafb', // gray-50
        },
        card: {
          border: '#e5e7eb', // gray-200
        }
      }
    },
  },
  plugins: [],
}
