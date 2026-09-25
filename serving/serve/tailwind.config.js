/* Compiled to a static stylesheet by the standalone Tailwind CLI on the box —
   no runtime JIT, so no flash of unstyled content on load. Rebuild with:
     serve/build-css.sh
   Theme tokens live here; components are in tw-input.css via @apply. */
module.exports = {
  content: ["./app.js", "./arena.html"],
  theme: {
    extend: {
      colors: {
        bg0:"#fbfbf9", surface:"#ffffff", panel:"#f6f7f5", raise:"#eef1ee",
        line:"#e3e7e3", line2:"#eef1ee",
        ink:"#17211d", soft:"#57665f", faint:"#93a29b",
        teal:"#0d9488", tealink:"#0f766e", tealsoft:"#ccfbf1",
        amber:"#b45309", ambersoft:"#fef0d9",
        red:"#dc2626", violet:"#7c3aed",
      },
      fontFamily: {
        mono: ['"Plex Mono"','ui-monospace','SFMono-Regular','Menlo','monospace'],
        sans: ['"Plex Sans"','ui-sans-serif','system-ui','sans-serif'],
      },
    },
  },
  corePlugins: { preflight: false },  // keep our own reset; avoid preflight stripping list bullets etc.
  plugins: [],
};
