// `npm run fidelity`: prints every Fidelity number measured from the fixtures (scripts/fidelity.ts), with its ceiling.
// Exits 1 when a measurement reaches its ceiling. Loads the TypeScript through Vite's module runner (a dev dependency
// through vitest), so the sources run as the tests run them, with nothing built.
import { createServer, createServerModuleRunner } from "vite";

const root = new URL("..", import.meta.url).pathname;
const server = await createServer({ root, configFile: false, logLevel: "error", appType: "custom", server: { middlewareMode: true, hmr: false, ws: false } });
const runner = createServerModuleRunner(server.environments.ssr, { hmr: false });
let fidelity, measured;
try {
  fidelity = await runner.import("/scripts/fidelity.ts");
  measured = await fidelity.measureFidelity();
} finally {
  await runner.close();
  await server.close();
}
console.log(fidelity.table(measured));
const over = fidelity.over(measured);
if (over.length) {
  console.error(`\nat or over the ceiling:\n${over.map((r) => `${r.quantity} (${r.conditions}): ${r.value} >= ${r.ceiling}`).join("\n")}`);
  process.exit(1);
}
