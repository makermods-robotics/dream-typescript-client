# Browser app through a proxy

The browser never holds a Dream API key. The page talks to `/api/dream` and `/api/dream-platform` on its own
origin, and `server.ts` forwards those calls to Dream Labs with the key added, the way the MakerMods console's
server does (`server/mm_server/dream_proxy.py` there).

```sh
npm run build                       # the SDK's dist/, which the page bundle imports
npx esbuild examples/browser-proxy/app.ts --bundle --format=esm --outfile=examples/browser-proxy/app.js
node examples/browser-proxy/server.ts   # Node 23.6+; reads ~/.dreamlabs/credentials or DREAM_* variables
open http://127.0.0.1:8787/
```

The page uses the webcam as the scene camera, asks Dream-1 Beta for SO-101 chunks with client-side IK, and prints
the rows instead of driving motors.

The page's client runs with `proxyHeaders: true`: it sends `Dream-Client-Version` and a random 8-hex
`Dream-Client-Instance`, and the proxy builds the gateway's `User-Agent: dream-client/<version> <instance>` from
them (a browser cannot set `User-Agent`). The proxy answers only requests addressed to it by name (`Host`, which
stops DNS rebinding) from its own page (`Origin`), and refuses bodies over 8 MiB.

**A deployed proxy needs authentication.** Add your own sign-in before it is reachable by anyone else: whoever
reaches it spends your key. The Host and Origin checks are not a sign-in.
