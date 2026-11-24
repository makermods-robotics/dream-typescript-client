# Browser app through a proxy

The browser never holds a Dream API key. The page's client is pointed at `/api/dream` on its own origin
(`new DreamClient({ proxyUrl: "/api/dream" })`), and `server.ts` serves `createDreamProxy()` there, which forwards the
client's calls to Dream Labs with the key added.

```sh
npm run build                       # the SDK's dist/, which the page bundle imports
npx esbuild examples/browser-proxy/app.ts --bundle --format=esm --outfile=examples/browser-proxy/app.js
node examples/browser-proxy/server.ts   # Node 23.6+; reads ~/.dreamlabs/credentials or DREAM_* variables
open http://127.0.0.1:8787/
```

The page uses the webcam as the scene camera, asks Dream-1 Beta for SO-101 chunks with client-side IK, and prints
the rows instead of driving motors.

The proxy forwards `get_action` and `end_session` to the gateway, the models list to the platform, and a fine-tune's
training data to the data service. It builds the gateway's `User-Agent: dream-client/<version> <instance>` from the
client's `Dream-Client-Version` and `Dream-Client-Instance` (a browser cannot set `User-Agent`), answers only
requests addressed to it by name (`allowedHosts`, which stops DNS rebinding) and, from a page, only its own
(`allowedOrigins`), and refuses bodies over 8 MiB.

Serving the page sets a random session cookie (`HttpOnly; SameSite=Strict`, so no other site's page, link or image
sends it), and the proxy's `authorize` forwards only requests that carry it.

**A deployed proxy needs authentication.** Add your own sign-in (`authorize`) before it is reachable by anyone else:
whoever reaches it spends your key. The host and origin checks are not a sign-in.
