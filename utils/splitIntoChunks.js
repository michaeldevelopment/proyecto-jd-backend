function splitIntoChunks(data) {
  let chunks = [];
  let currentChunk = [];

  for (let i = 0; i < data.length; i++) {
    currentChunk.push(data[i]);

    if (data[i].hora === 23 && data[i + 1] && data[i + 1].hora === 0) {
      chunks.push(currentChunk);
      currentChunk = [];
    }
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk);
  }

  return chunks;
}

module.exports = splitIntoChunks;
