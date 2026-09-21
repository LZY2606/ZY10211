package maxminddb

// Differential tests that pin the boundary between eager validation and lazy
// decoding.
//
// Semantics
//
// Every database starts as a structurally valid, minimal MMDB assembled by the
// in-test builders below (see newMinimalFixture). Tests never flip random
// bytes: each mutation changes one declared field with its format meaning,
// e.g. the metadata node_count, a search-tree record value, the separator, a
// container length header, a shared data pointer, or a UTF-8 payload.
//
// For each mutation the same byte slice is driven through every public path:
// OpenBytes, Reader.Verify, Reader.Lookup plus Result.Decode,
// Result.DecodePath, and a capped Reader.Networks traversal. Open is lazy on
// purpose - it decodes metadata and locates sections but does not walk the
// tree or data section - so cases where Open succeeds and Verify rejects are
// expected and are recorded in the test log as "lazy-open-allowed" cases.
//
// Complexity
//
// The fixtures are constant-size (at most ~1 KiB). Each test operation is
// bounded explicitly: decoding keeps the library's own limits (32,768
// declared container child slots and a 2 MiB materialized payload allowance
// per operation), and the test harness additionally caps Networks iteration
// (networksIterationCap results). A corrupt search tree is at most 128 levels
// deep; cycle and fan-out cases are therefore reached in constant work, and
// the fan-out case must be rejected by the decoder budget rather than walked.
//
// Compatibility trade-offs
//
// The tests assert current public behavior (lazy Open, eager Verify) and use
// only public APIs plus in-memory bytes; they do not require files, network
// access, the wall clock, or any particular directory order. Invalid UTF-8 is
// tolerated by lazy decoding (bytes are copied as declared) but rejected by
// Verify; tests lock that trade-off instead of forcing validation into the
// lookup path, which would cost throughput on every read.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// Search-tree record references. A node pair stores two refs that the
// assembler resolves into raw record values: an empty record (node_count), a
// node index, or a data-section label (node_count + 16 + label offset).
type refKind uint8

const (
	refEmpty refKind = iota
	refNode
	refData
)

type nodeRef struct {
	kind refKind
	idx  uint
	leaf string
}

func emptyRef() nodeRef { return nodeRef{kind: refEmpty} }
func nodeRefTo(i uint) nodeRef {
	return nodeRef{kind: refNode, idx: i}
}
func dataRef(label string) nodeRef {
	return nodeRef{kind: refData, leaf: label}
}

// dataWriter assembles one sequential data-section chunk. Labels name
// control-byte offsets so mutations can retarget pointers or patch headers by
// semantics instead of hard-coding byte positions.
type dataWriter struct {
	buf    []byte
	labels map[string]uint
}

func newDataWriter() *dataWriter {
	return &dataWriter{labels: map[string]uint{}}
}

func (w *dataWriter) label(name string) {
	w.labels[name] = uint(len(w.buf))
}

// ctrl appends a non-pointer control header for kind with the declared size.
func (w *dataWriter) ctrl(kind, size uint) {
	switch {
	case size < 29:
		w.buf = append(w.buf, byte(kind<<5|size))
	case size < 285:
		w.buf = append(w.buf,
			byte(kind<<5|29),
			byte(size-29),
		)
	case size < 65821:
		v := size - 285
		w.buf = append(w.buf,
			byte(kind<<5|30),
			byte(v>>8), byte(v),
		)
	default:
		v := size - 65821
		w.buf = append(w.buf,
			byte(kind<<5|31),
			byte(v>>16), byte(v>>8), byte(v),
		)
	}
}

func (w *dataWriter) stringToken(name, value string) {
	if name != "" {
		w.label(name)
	}
	w.ctrl(uint(KindStringForTest), uint(len(value)))
	w.buf = append(w.buf, value...)
}

func (w *dataWriter) uint16Token(v uint16) {
	// KindUint16 (5), size 2: ctrl 0xA2 followed by two big-endian bytes.
	w.buf = append(w.buf, 0xA2, byte(v>>8), byte(v))
}

// replaceAtLabel replaces length bytes starting at a labeled control byte with
// repl, shifting every later label by the size delta. It models semantic
// header rewrites (e.g. widening a length field) rather than random flips.
func (w *dataWriter) replaceAtLabel(name string, length int, repl []byte) {
	at, ok := w.labels[name]
	if !ok {
		panic("mutation test label missing: " + name)
	}
	delta := len(repl) - length
	newBuf := make([]byte, 0, len(w.buf)+delta)
	newBuf = append(newBuf, w.buf[:at]...)
	newBuf = append(newBuf, repl...)
	newBuf = append(newBuf, w.buf[at+uint(length):]...)
	w.buf = newBuf
	for k, off := range w.labels {
		if off > at {
			w.labels[k] = off + uint(delta)
		}
	}
}

// ptrToken appends a pointer to a label using the narrowest legal encoding.
// The target is patched while assembling, after every label is known.
func (w *dataWriter) ptrToken(targetLabel string) {
	w.label(targetLabel + ":ptr")
	w.buf = append(w.buf, 0x20, 0x00) // width-one placeholder
}

func encodePointer(buf []byte, at, target uint) {
	switch {
	case target < 2048:
		buf[at] = 0x20 | byte((target>>8)&0x07)
		buf[at+1] = byte(target)
	case target < 526336:
		buf[at] = 0x28 | byte((target>>16)&0x07)
		buf[at+1] = byte(target >> 8)
		buf[at+2] = byte(target)
	default:
		buf[at] = 0x38
		binary.BigEndian.PutUint32(buf[at+1:at+5], uint32(target))
	}
}

// KindStringForTest mirrors decoder.KindString (2) without reaching into the
// internal package; the value is fixed by the MMDB spec.
const KindStringForTest = 2

// pointerBase2ForTest mirrors the decoder's width-two pointer base (2048).
const pointerBase2ForTest = 2048

// metaField is one encoded metadata map field.
type metaField struct {
	name  string
	value []byte
}

func metaKey(name string) []byte {
	if len(name) > 28 {
		panic("mutation test metadata key too long: " + name)
	}
	out := []byte{byte(KindStringForTest<<5 | uint(len(name)))}
	return append(out, name...)
}

func metaU16(v uint) []byte {
	// KindUint16 (high three bits 5) with minimal big-endian value width.
	switch {
	case v <= 0xFF:
		return []byte{0xA1, byte(v)}
	case v <= 0xFFFF:
		return []byte{0xA2, byte(v >> 8), byte(v)}
	default:
		panic(fmt.Sprintf("mutation test u16 overflow: %d", v))
	}
}

func metaU32(v uint) []byte {
	// KindUint32 (high three bits 6) with minimal big-endian value width.
	switch {
	case v <= 0xFF:
		return []byte{0xC1, byte(v)}
	case v <= 0xFFFF:
		return []byte{0xC2, byte(v >> 8), byte(v)}
	case v <= 0xFFFFFF:
		return []byte{0xC3, byte(v >> 16), byte(v >> 8), byte(v)}
	default:
		return []byte{0xC4, byte(v >> 24), byte(v >> 16), byte(v >> 8), byte(v)}
	}
}
func metaU64(v uint64) []byte {
	// Extended kind: low-five size byte, then kind-7 (Uint64-7 = 2).
	b := []byte{0x08, 0x02, 0, 0, 0, 0, 0, 0, 0, 0}
	binary.BigEndian.PutUint64(b[2:], v)
	return b
}
func metaString(v string) []byte {
	if len(v) > 28 {
		panic("mutation test metadata string too long: " + v)
	}
	out := []byte{byte(KindStringForTest<<5 | uint(len(v)))}
	return append(out, v...)
}
func metaEmptyArray() []byte {
	// KindSlice is 11, an extended kind: a low-five size byte (0) is
	// followed by the extended kind byte (11-7 = 4).
	return []byte{0x00, 0x04}
}
func metaDescription() []byte {
	// map size 1: "en" -> "mut"
	return []byte{
		0xE1,
		0x42, 'e', 'n',
		0x43, 'm', 'u', 't',
	}
}

// fixtureSpec is the semantic, pre-assembly description of a database.
type fixtureSpec struct {
	name       string
	ipVersion  uint
	recordSize uint
	nodes      [][2]nodeRef
	// data chunks in linear order; labels resolve across the concatenation.
	chunks []*dataWriter
	// leafPaths maps a data label at a chunk start to DecodePath probes.
	leafPaths map[string][]probePath
	separator [16]byte
	markerOK  bool
	truncate  int // absolute final length; 0 means no truncation
	fields    []metaField
	// rawPatch performs a semantic byte patch after layout is known but
	// before truncation; it is used for tree/data corruption mutations.
	rawPatch func(t *testing.T, db *assembledDB)
}

type probePath struct {
	name  string
	elems []any
}

// minimalFields returns metadata fields in a fixed order. node_count is last
// so truncation mutations can cut a varint off the end of the file while
// leaving the marker and all earlier fields intact.
func minimalFields(nodeCount, recordSize, ipVersion uint) []metaField {
	return []metaField{
		{"binary_format_major_version", metaU16(2)},
		{"binary_format_minor_version", metaU16(0)},
		{"record_size", metaU16(recordSize)},
		{"ip_version", metaU16(ipVersion)},
		{"database_type", metaString("Mutation-Test")},
		{"description", metaDescription()},
		{"build_epoch", metaU64(1_700_000_000)},
		{"languages", metaEmptyArray()},
		{"node_count", metaU32(nodeCount)},
	}
}

func (f *fixtureSpec) setMeta(name string, value []byte) {
	for i := range f.fields {
		if f.fields[i].name == name {
			f.fields[i].value = value
			return
		}
	}
	f.fields = append(f.fields, metaField{name: name, value: value})
}

// refRaw forces an exact raw record value (24/28/32-bit as assembled),
// bypassing ref resolution so tree-corruption mutations can plant values.
const refRaw refKind = 99

// assembledDB is a built database plus everything the harness needs to
// interpret successful results against declared structure.
type assembledDB struct {
	buf         []byte
	nodeCount   uint
	recordSize  uint
	dataStart   uint
	dataLen     uint
	labelOffset map[string]uint
}

// dataSection returns the declared data section window. Mutations that change
// declared node_count or the file length can make the window exceed the
// buffer; in that case the usable suffix (possibly empty) is returned.
func (d assembledDB) dataSection() []byte {
	n := uint(len(d.buf))
	if d.dataStart >= n {
		return nil
	}
	end := d.dataStart + d.dataLen
	if end > n {
		end = n
	}
	return d.buf[d.dataStart:end]
}

func (f *fixtureSpec) assemble(t *testing.T) assembledDB {
	t.Helper()

	nodeCount := uint(len(f.nodes))

	var data []byte
	labelOffset := map[string]uint{}
	for _, chunk := range f.chunks {
		base := uint(len(data))
		for name, off := range chunk.labels {
			labelOffset[name] = base + off
		}
		data = append(data, chunk.buf...)
	}

	recordValue := func(r nodeRef) uint {
		switch r.kind {
		case refEmpty:
			return nodeCount
		case refNode:
			return r.idx
		case refRaw:
			return r.idx
		default:
			off, ok := labelOffset[r.leaf]
			require.True(t, ok, "unknown data label %q", r.leaf)
			return nodeCount + dataSectionSeparatorSize + off
		}
	}

	mult := f.recordSize / 4
	tree := make([]byte, nodeCount*mult)
	for i, pair := range f.nodes {
		base := uint(i) * mult
		writeRecordPair(t, f.recordSize, tree, base,
			recordValue(pair[0]), recordValue(pair[1]))
	}

	// Resolve data-section pointer placeholders now that targets are known.
	for _, chunk := range f.chunks {
		base := chunkBase(f, chunk)
		for name := range chunk.labels {
			if len(name) < 4 || name[len(name)-4:] != ":ptr" {
				continue
			}
			targetName := name[:len(name)-4]
			target, ok := labelOffset[targetName]
			require.True(t, ok, "pointer target %q missing", targetName)
			encodePointer(data, base+chunk.labels[name], target)
		}
	}

	fields := make([]byte, 0, 128)
	for _, field := range f.fields {
		fields = append(fields, metaKey(field.name)...)
		fields = append(fields, field.value...)
	}
	require.Less(t, len(f.fields), 29)
	metadata := append([]byte{0xE0 | byte(len(f.fields))}, fields...)

	marker := metadataStartMarker
	if !f.markerOK {
		marker = bytes.Clone(metadataStartMarker)
		marker[0] = 0x00
	}

	buf := make([]byte, 0, len(tree)+16+len(data)+len(marker)+len(metadata))
	buf = append(buf, tree...)
	buf = append(buf, f.separator[:]...)
	buf = append(buf, data...)
	buf = append(buf, marker...)
	buf = append(buf, metadata...)

	db := assembledDB{
		buf:         buf,
		nodeCount:   nodeCount,
		recordSize:  f.recordSize,
		dataStart:   nodeCount*mult + dataSectionSeparatorSize,
		dataLen:     uint(len(data)),
		labelOffset: labelOffset,
	}
	if f.rawPatch != nil {
		f.rawPatch(t, &db)
	}
	if f.truncate > 0 && f.truncate < len(db.buf) {
		db.buf = db.buf[:f.truncate]
	}
	return db
}

func chunkBase(f *fixtureSpec, want *dataWriter) uint {
	var base uint
	for _, chunk := range f.chunks {
		if chunk == want {
			return base
		}
		base += uint(len(chunk.buf))
	}
	return base
}

func writeRecordPair(
	t *testing.T,
	recordSize uint,
	buf []byte,
	base, left, right uint,
) {
	t.Helper()
	switch recordSize {
	case 24:
		buf[base] = byte(left >> 16)
		buf[base+1] = byte(left >> 8)
		buf[base+2] = byte(left)
		buf[base+3] = byte(right >> 16)
		buf[base+4] = byte(right >> 8)
		buf[base+5] = byte(right)
	case 28:
		buf[base] = byte(left >> 16)
		buf[base+1] = byte(left >> 8)
		buf[base+2] = byte(left)
		buf[base+3] = byte(((left>>20)&0x0F)<<4 | (right>>24)&0x0F)
		buf[base+4] = byte(right >> 16)
		buf[base+5] = byte(right >> 8)
		buf[base+6] = byte(right)
	case 32:
		binary.BigEndian.PutUint32(buf[base:base+4], uint32(left))
		binary.BigEndian.PutUint32(buf[base+4:base+8], uint32(right))
	default:
		t.Fatalf("unsupported record size %d", recordSize)
	}
}

// buildLeafA assembles the richer tree-referenced chunk ("leafA"):
//
//	map size 5
//	  "name"       -> pointer to "shared"
//	  "count"      -> uint16 42
//	  "shared"     -> string "en" (shared target; appears in linear flow)
//	  "nested"     -> map size 1 { "city" -> string "Tokyo" }
//	  "tags"       -> array size 3 [1, 2, 3]
//
// Every byte is within the linear footprint of this chunk (no forward
// references into trailing data), so Verify accepts it.
func buildLeafA() *dataWriter {
	w := newDataWriter()
	w.label("leafA")
	w.ctrl(7, 5) // KindMap
	w.stringToken("keyName", "name")
	w.ptrToken("shared")
	w.stringToken("keyCount", "count")
	w.uint16Token(42)
	w.stringToken("keyShared", "shared")
	w.stringToken("shared", "en")
	w.stringToken("keyNested", "nested")
	w.label("nestedMap")
	w.ctrl(7, 1)
	w.stringToken("keyCity", "city")
	w.stringToken("cityTokyo", "Tokyo")
	w.stringToken("keyTags", "tags")
	w.label("tagsArr")
	// KindSlice (11) is an extended kind: size byte followed by kind-7=4.
	w.buf = append(w.buf, 0x03, 0x04) // array of 3
	w.uint16Token(1)
	w.uint16Token(2)
	w.uint16Token(3)
	return w
}

// buildLeafB assembles the second tree-referenced chunk ("leafB"):
//
//	map size 1 { "name" -> string "ja" }
func buildLeafB() *dataWriter {
	w := newDataWriter()
	w.label("leafB")
	w.ctrl(7, 1)
	w.stringToken("keyBName", "name")
	w.stringToken("bName", "ja")
	return w
}

func leafProbes() map[string][]probePath {
	return map[string][]probePath{
		"leafA": {
			{name: "name", elems: []any{"name"}},
			{name: "count", elems: []any{"count"}},
			{name: "nested_city", elems: []any{"nested", "city"}},
			{name: "tags_first", elems: []any{"tags", 0}},
			{name: "tags_last", elems: []any{"tags", -1}},
		},
		"leafB": {
			{name: "name", elems: []any{"name"}},
		},
	}
}

// newMinimalFixture builds a valid minimal IPv4 database with a three-level
// search tree:
//
//	node 0: bit 0 = 0 -> node 1, bit 0 = 1 -> empty node 2
//	node 1: bit 1 = 0 -> leafA, bit 1 = 1 -> leafB
//	nodes 2 and 3: empty padding
//
// so 0.0.0.0/2 (includes 10.0.0.1) and 64.0.0.0/2 (includes 64.0.0.1)
// carry the two data records, while 128.0.0.0/1 is empty.
func newMinimalFixture(recordSize uint) *fixtureSpec {
	nodes := make([][2]nodeRef, 4)
	// bit 0 = 0 -> node 1 (covers 0.0.0.0/1: 10.x starts 00, 64.x starts 01)
	// bit 0 = 1 -> node 2 (unused, empty)
	nodes[0] = [2]nodeRef{nodeRefTo(1), nodeRefTo(2)}
	// bit 1 = 0 -> leafA (00 = 0.0.0.0/2, includes 10.0.0.1)
	// bit 1 = 1 -> leafB (01 = 64.0.0.0/2, includes 64.0.0.1)
	nodes[1] = [2]nodeRef{dataRef("leafA"), dataRef("leafB")}
	nodes[2] = [2]nodeRef{emptyRef(), emptyRef()}
	nodes[3] = [2]nodeRef{emptyRef(), emptyRef()}
	return &fixtureSpec{
		name:       fmt.Sprintf("minimal-ipv4-%d", recordSize),
		ipVersion:  4,
		recordSize: recordSize,
		nodes:      nodes,
		chunks:     []*dataWriter{buildLeafA(), buildLeafB()},
		leafPaths:  leafProbes(),
		markerOK:   true,
		fields:     minimalFields(4, recordSize, 4),
	}
}

// newIPv6Fixture builds a valid IPv6 database whose ::/0 root walks an
// all-zero chain for 96 bits. The node reached after those bits (node 96)
// is the IPv4 subtree root and mirrors the IPv4 fixture's two-level tree:
//
//	node 96 (ipv4 start): bit 0 = 0 -> node 97, bit 0 = 1 -> empty node 98
//	node 97: bit 1 = 0 -> leafA (00 covers 10.0.0.1), bit 1 = 1 -> leafB
//	         (01 covers 64.0.0.1)
//	node 98: unused padding
//
// The IPv4 start node is therefore node 96, the target of the IPv4-start
// corruption mutations.
func newIPv6Fixture() *fixtureSpec {
	const count = 99
	nodes := make([][2]nodeRef, count)
	for i := range nodes {
		nodes[i] = [2]nodeRef{emptyRef(), emptyRef()}
	}
	// All-zero path for the first 96 bits: node i zero record -> node i+1.
	for i := uint(0); i < 96; i++ {
		nodes[i] = [2]nodeRef{nodeRefTo(i + 1), emptyRef()}
	}
	nodes[96] = [2]nodeRef{nodeRefTo(97), emptyRef()}
	nodes[97] = [2]nodeRef{dataRef("leafA"), dataRef("leafB")}
	// node 98 is unused padding.
	return &fixtureSpec{
		name:       "minimal-ipv6-24",
		ipVersion:  6,
		recordSize: 24,
		nodes:      nodes,
		chunks:     []*dataWriter{buildLeafA(), buildLeafB()},
		leafPaths:  leafProbes(),
		markerOK:   true,
		fields:     minimalFields(count, 24, 6),
	}
}

// newFanoutFixture builds a single referenced chunk whose root is a chain
// of 16 nested two-element arrays. Each array's two pointers target the
// previous (one level shallower) array, and the deepest array points at a
// single uint16 leaf at the tail. Following every child would cost
// 2**16 visits; both Verify and lazy Decode must reject it through the
// declared-child budget instead of materializing the fan-out. The root is
// at offset 0 so the verifier's sequential scan reaches it as the first
// (and only) tree-referenced value.
func newFanoutFixture() *fixtureSpec {
	const depth = 16
	buf := make([]byte, 0, depth*8+3)
	arrayStart := make([]uint, depth)
	for level := depth - 1; level >= 0; level-- {
		arrayStart[level] = uint(len(buf))
		buf = append(buf, 0x02, 0x04) // extended KindSlice, size 2
		buf = append(buf, 0, 0)       // first width-one pointer slot
		buf = append(buf, 0, 0)       // second width-one pointer slot
	}
	leafOff := uint(len(buf))
	buf = append(buf, 0xA2, 0x00, 0x00) // uint16 0 leaf at the tail

	putPointer := func(at, target uint) {
		// All targets are below 2048, so the width-one encoding (base 0)
		// applies: high five control bits 00100 and the target in the low
		// three control bits plus one following byte.
		buf[at] = 0x20 | byte((target>>8)&0x07)
		buf[at+1] = byte(target)
	}
	for level := range depth {
		target := leafOff
		if level > 0 {
			target = arrayStart[level-1]
		}
		putPointer(arrayStart[level]+2, target)
		putPointer(arrayStart[level]+4, target)
	}

	chunk := &dataWriter{
		buf:    buf,
		labels: map[string]uint{"fanRoot": arrayStart[depth-1]},
	}
	nodes := [][2]nodeRef{
		{dataRef("fanRoot"), emptyRef()},
	}
	return &fixtureSpec{
		name:       "fanout-ipv4-24",
		ipVersion:  4,
		recordSize: 24,
		nodes:      nodes,
		chunks:     []*dataWriter{chunk},
		leafPaths:  map[string][]probePath{},
		markerOK:   true,
		fields:     minimalFields(1, 24, 4),
	}
}
