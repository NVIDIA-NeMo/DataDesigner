/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        nvidia: {
          green: "#76b900",
          "green-dark": "#5a8f00",
          "green-light": "#8ed100",
        },
        surface: {
          0: "#0a0a0a",
          1: "#141414",
          2: "#1e1e1e",
          3: "#282828",
          4: "#323232",
        },
        border: {
          DEFAULT: "#333333",
          hover: "#444444",
          focus: "#76b900",
        },
      },
    },
  },
  plugins: [],
};
