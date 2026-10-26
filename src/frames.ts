/**
 * Camera frames for the window: each one a base64 JPEG string, resized and encoded by the caller (the client never
 * decodes one). Helpers for the two ends people have: raw JPEG bytes (Node, a camera library) and a browser image
 * source (a `<video>`, an `ImageBitmap`, a `VideoFrame`), drawn on an `OffscreenCanvas`.
 */

import { DreamConfigError } from "./errors.js";

/** The size and quality the MakerMods console sends Dream-1: 320 x 176 at JPEG quality 0.85. */
export const CONSOLE_FRAME = { width: 320, height: 176, quality: 0.85 } as const;

/** Base64 of bytes (a JPEG file, say), as a frame. */
export function bytesToBase64(bytes: ArrayBuffer | ArrayBufferView): string {
  const view = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let binary = "";
  for (let i = 0; i < view.length; i += 0x8000) binary += String.fromCharCode(...view.subarray(i, i + 0x8000));
  return btoa(binary);
}

/**
 * One camera image as a frame: drawn (stretched) to `width` x `height` on an `OffscreenCanvas`, encoded as JPEG, then
 * base64. Browsers and workers only (Deno and Bun lack `OffscreenCanvas`).
 *
 * @throws DreamConfigError There is no `OffscreenCanvas` here.
 */
export async function encodeJpegFrame(
  source: CanvasImageSource,
  options: { width?: number; height?: number; quality?: number } = {},
): Promise<string> {
  if (typeof OffscreenCanvas === "undefined") {
    throw new DreamConfigError("encodeJpegFrame() needs OffscreenCanvas (a browser or a worker); elsewhere encode the JPEG yourself and use bytesToBase64()");
  }
  const { width = CONSOLE_FRAME.width, height = CONSOLE_FRAME.height, quality = CONSOLE_FRAME.quality } = options;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  if (!context) throw new DreamConfigError("encodeJpegFrame(): no 2D context on an OffscreenCanvas");
  context.drawImage(source, 0, 0, width, height);
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality });
  return bytesToBase64(await blob.arrayBuffer());
}
