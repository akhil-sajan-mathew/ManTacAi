/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                midnight: "#050510",
                "glass-panel": "rgba(20, 25, 40, 0.6)",
                "glass-border": "rgba(100, 200, 255, 0.08)",
                "neon-purple": "#a855f7",
                "neon-cyan": "#06b6d4",
                "neon-red": "#f43f5e",
                "neon-green": "#10b981",
            },
            fontFamily: {
                inter: ['Inter', 'sans-serif'],
                orbitron: ['Orbitron', 'sans-serif'],
            },
            backgroundImage: {
                'cyber-gradient': 'radial-gradient(circle at 50% 10%, #1a1a40 0%, #050510 60%)',
            }
        },
    },
    plugins: [],
}
