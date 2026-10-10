/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Chart chrome & ink from the validated dark palette (dataviz skill).
        surface: "#1a1a19",
        plane: "#0d0d0d",
        ink: "#f0efec",
        "ink-secondary": "#c3c2b7",
        muted: "#898781",
        grid: "#2c2c2a",
        baseline: "#383835",
        // Categorical series slots (dark steps).
        "series-1": "#3987e5",
        "series-2": "#d95926",
        "series-3": "#199e70",
        "series-4": "#c98500",
        // Status (fixed, never themed).
        good: "#0ca30c",
        warning: "#fab219",
        critical: "#d03b3b",
      },
    },
  },
  plugins: [],
};
