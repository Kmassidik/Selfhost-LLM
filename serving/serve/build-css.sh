#!/bin/bash
# Compile the static stylesheet. Run after changing classes in app.js or tw-input.css.
cd /root/Desktop/selfhostllm/serve
/tmp/tw -c tailwind.config.js -i tw-input.css -o arena.css --minify
echo "built arena.css ($(wc -c < arena.css) bytes)"
