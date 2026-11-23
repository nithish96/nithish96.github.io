import { cpSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const katexDist = fileURLToPath(new URL("../node_modules/katex/dist/", import.meta.url));
const katexAssets = fileURLToPath(new URL("../docs/assets/katex/", import.meta.url));

mkdirSync(join(katexAssets, "contrib"), { recursive: true });
cpSync(join(katexDist, "katex.min.js"), join(katexAssets, "katex.min.js"));
cpSync(join(katexDist, "katex.min.css"), join(katexAssets, "katex.min.css"));
cpSync(
  join(katexDist, "contrib", "auto-render.min.js"),
  join(katexAssets, "contrib", "auto-render.min.js"),
);
cpSync(join(katexDist, "fonts"), join(katexAssets, "fonts"), { recursive: true });
