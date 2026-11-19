// Four preintegrated views share one texture so scene selection has no GPU start-up cost.
const scenes = ['wood', 'pool', 'jet', 'crosswind'];
const atlas = new Image();

function drawPreviews() {
  const width = atlas.naturalWidth;
  const height = atlas.naturalHeight / scenes.length;
  for (const canvas of document.querySelectorAll('canvas[data-thumb]')) {
    const index = scenes.indexOf(canvas.dataset.thumb);
    if (index < 0) continue;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { alpha: false });
    context?.drawImage(atlas, 0, index * height, width, height, 0, 0, width, height);
  }
}

atlas.addEventListener('load', drawPreviews, { once: true });
atlas.src = new URL('./scenario-atlas.webp', import.meta.url).href;
