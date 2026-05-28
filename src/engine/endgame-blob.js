// Packed-key codec for the endgame DB blob.
//
// The blob is a flat, fixed-stride array of (key, value) entries sorted by
// packed-byte order. Lookup is a fixed-stride binary search — no parsing
// at lookup time.
//
// Per-entry layout for a blob with parameter N_MAX:
//   byte 0    high 4 bits = s0 length (0..10), low 4 bits = s1 length (0..11)
//   byte 1    high 4 bits = s2 length (0..11), low 4 bits = myTricks (0..13)
//   byte 2    high 4 bits = oppTricks (0..13), low 4 bits unused
//   bytes 3+  packed location codes, 2 bits each, big-endian within bytes
//   last byte int8 value (signed margin, range -6..+6)
//
// Location codes: '0' (mover) → 0b00, '1' (opp) → 0b01, '2' (led card) → 0b10.
// Lower-depth entries pad unused location bits with zeros so all entries
// share the same byte width.
//
// File header (16 bytes total, prefixed once at the start of the blob):
//   bytes 0..3   magic ASCII 'FXEG'
//   bytes 4..5   uint16 LE version (currently 1)
//   bytes 6..7   reserved (must be 0)
//   bytes 8..11  uint32 LE N_MAX
//   bytes 12..15 uint32 LE entry count

export const HEADER_BYTES = 3
export const FILE_HEADER_BYTES = 16
export const BLOB_MAGIC = 0x47455846 // 'FXEG' little-endian as uint32
export const BLOB_VERSION = 1

export function locationBytesForNMax(nMax) {
  return Math.ceil((2 * nMax) / 4)
}

export function keyByteWidth(nMax) {
  return HEADER_BYTES + locationBytesForNMax(nMax)
}

export function entryByteWidth(nMax) {
  return keyByteWidth(nMax) + 1 // +1 for int8 value
}

// Pack a canonical-key string into `out` at `offset`. Writes exactly
// keyByteWidth(nMax) bytes. `locationBytes = locationBytesForNMax(nMax)`.
export function packCanonKey(canonKey, out, offset, locationBytes) {
  // Format: "s0|s1|s2|myTricks,oppTricks"
  const p1 = canonKey.indexOf('|')
  const p2 = canonKey.indexOf('|', p1 + 1)
  const p3 = canonKey.indexOf('|', p2 + 1)
  const s0 = canonKey.slice(0, p1)
  const s1 = canonKey.slice(p1 + 1, p2)
  const s2 = canonKey.slice(p2 + 1, p3)
  const ticksPart = canonKey.slice(p3 + 1)
  const comma = ticksPart.indexOf(',')
  const myT = +ticksPart.slice(0, comma)
  const oppT = +ticksPart.slice(comma + 1)

  out[offset] = ((s0.length & 0xf) << 4) | (s1.length & 0xf)
  out[offset + 1] = ((s2.length & 0xf) << 4) | (myT & 0xf)
  out[offset + 2] = (oppT & 0xf) << 4

  for (let i = 0; i < locationBytes; i++) out[offset + HEADER_BYTES + i] = 0

  let bitPos = 0
  const writeLoc = (charCode) => {
    const v = charCode - 48 // '0' → 0, '1' → 1, '2' → 2
    const byteIdx = offset + HEADER_BYTES + (bitPos >> 3)
    const shift = 6 - (bitPos & 7)
    out[byteIdx] |= v << shift
    bitPos += 2
  }
  for (let i = 0; i < s0.length; i++) writeLoc(s0.charCodeAt(i))
  for (let i = 0; i < s1.length; i++) writeLoc(s1.charCodeAt(i))
  for (let i = 0; i < s2.length; i++) writeLoc(s2.charCodeAt(i))
}

// Fixed-stride binary search over packed entries. Returns int8 value if
// found, null otherwise.
//
//   data       Uint8Array of length entryCount * entryWidth
//   keyBytes   Uint8Array of length keyWidth holding the query key
//   keyWidth   total key bytes (header + location)
//   entryWidth keyWidth + 1
//   count      number of entries
export function lookupPacked(data, keyBytes, keyWidth, entryWidth, count) {
  let lo = 0
  let hi = count - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const base = mid * entryWidth
    let cmp = 0
    for (let i = 0; i < keyWidth; i++) {
      const a = data[base + i]
      const b = keyBytes[i]
      if (a !== b) {
        cmp = a - b
        break
      }
    }
    if (cmp === 0) {
      const v = data[base + keyWidth]
      return v < 128 ? v : v - 256 // int8 from uint8
    }
    if (cmp < 0) lo = mid + 1
    else hi = mid - 1
  }
  return null
}
