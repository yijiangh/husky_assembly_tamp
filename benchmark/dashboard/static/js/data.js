// Fetching and caching the benchmark's files.
//
// A batch is far too big to load at once, so it is written as a tree of small
// files and the page pulls in whichever branch it is showing. Each file is
// fetched at most once per session.

const cache = new Map();

export async function loadJson(url) {
  if (cache.has(url)) return cache.get(url);
  const promise = fetch(url).then((response) => {
    if (!response.ok) {
      throw new Error(`could not load ${url} (HTTP ${response.status})`);
    }
    return response.json();
  });
  cache.set(url, promise);
  return promise;
}

// Reads a gzipped JSON file, unpacking it in the browser.
//
// Python's built-in web server sends a .gz file as plain bytes with no header
// telling the browser it is compressed, so the page unpacks it itself. The
// failed-attempt files use this: a coarse pass has a couple of thousand
// attempts, which gzip shrinks about six times.
export async function loadJsonGz(url) {
  if (cache.has(url)) return cache.get(url);
  const promise = fetch(url).then(async (response) => {
    if (!response.ok) {
      throw new Error(`could not load ${url} (HTTP ${response.status})`);
    }
    if (typeof DecompressionStream === "undefined") {
      throw new Error("this browser cannot unpack gzip files "
        + "(no DecompressionStream); use a recent Chrome, Edge or Firefox");
    }
    const unpacked = response.body.pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(unpacked).text());
  });
  cache.set(url, promise);
  return promise;
}

// Reads a planned motion from its binary file.
//
// The format is four magic bytes, then version, waypoint count and joint count
// as 32-bit integers, then the joint angles as 32-bit floats. A wrong or
// truncated file is rejected rather than drawn as nonsense.
export async function loadTrajectory(url) {
  if (cache.has(url)) return cache.get(url);
  const promise = fetch(url).then(async (response) => {
    if (!response.ok) {
      throw new Error(`could not load ${url} (HTTP ${response.status})`);
    }
    const buffer = await response.arrayBuffer();
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 4));
    if (magic !== "HTRJ") {
      throw new Error(`${url} is not a trajectory file`);
    }
    const header = new Uint32Array(buffer, 4, 3);
    const [, waypoints, joints] = header;
    const expected = 16 + waypoints * joints * 4;
    if (buffer.byteLength < expected) {
      throw new Error(
        `${url} is truncated: expected ${expected} bytes, got ${buffer.byteLength}`
      );
    }
    const values = new Float32Array(buffer, 16, waypoints * joints);
    const path = [];
    for (let i = 0; i < waypoints; i += 1) {
      path.push(Array.from(values.subarray(i * joints, (i + 1) * joints)));
    }
    return path;
  });
  cache.set(url, promise);
  return promise;
}

// Where each kind of file lives, relative to the dashboard root.
export const paths = {
  run: () => "run.json",
  summary: () => "summary.json",
  // Comparisons of several batches live at the data root, beside runs.json,
  // because they belong to no single batch.
  compareIndex: () => "../../../compare/index.json",
  compare: (name) => `../../../compare/${name}.json`,
  bar: (barId) => `bars/${barId}.json`,
  goal: (barId, goalIndex) =>
    `bars/${barId}/goal${String(goalIndex).padStart(3, "0")}.json`,
  start: (barId, goalIndex, startIndex) =>
    `bars/${barId}/goal${String(goalIndex).padStart(3, "0")}/` +
    `start${String(startIndex).padStart(4, "0")}.json`,
};
