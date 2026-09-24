package maxminddb

import (
	"net/netip"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// Structured-mutation differential tests.
//
// These tests build a minimal but legal MMDB entirely in memory and then apply
// field-aware mutations (not random byte flips): metadata node count and
// record size, the metadata marker, search-tree pointers, the data-section
// separator, map/array length headers, shared data pointers, and UTF-8
// payloads. Every mutated database is exercised through Open/OpenBytes,
// Verify, Lookup+Decode, DecodePath, and a bounded Networks walk so that the
// boundary between lazy opening and strict verification is locked:
//
//  1. A database may open lazily and fail only at a later stage; that is
//     recorded, not treated as a contradiction.
//  2. Any successful Decode/DecodePath result stays inside the declared data
//     section and inside the decoder's documented operation budget
//     (32,768 container child slots and 2 MiB of materialized payload).
//  3. For legal databases, DecodePath leaves equal a full Decode followed by
//     walking the same path. For corrupt databases both paths may fail at
//     different times, but one path must not return data while the other
//     reports a structural error.
//  4. If Verify succeeds, lookups, decodes, and bounded network iteration
//     must not produce structural errors.
//  5. Duplicate map keys are the one documented value-level divergence:
//     Decode into a Go map keeps the last entry while DecodePath returns the
//     first match. The verifier currently accepts such databases. The
//     duplicate-map-key mutation locks this boundary explicitly; value
//     equivalence is only asserted for fixtures built without duplicate keys.
//
// All budgets below are explicit so a crafted fixture cannot make the test
// itself loop or allocate without limit.

const (
	// Mirrors the reflection decoder's per-operation child-slot limit.
	mutationMaxContainerSlots = 32_768
	// Mirrors the reflection decoder's exact per-operation payload allowance.
	mutationMaxPayloadBytes = 2 << 20
	// Maximum number of Networks results consumed per table-driven mutation.
	mutationMaxNetworks = 64
	// Smaller cap for fuzz cases; fuzzed input is also size-capped.
	mutationMaxFuzzNetworks = 16
	// Keeps the test-side recursive measure of decoded values bounded even
	// if a future decoder relaxed its own depth guard.
	mutationMaxMeasureDepth = 1024
	// Upper bound on bytes handed to a single fuzz case. Verify walks every
	// node and referenced record, so this keeps worst-case work bounded.
	mutationMaxFuzzSize = 8 << 10
)

// MMDB data type numbers used by the in-test encoder.
const (
	mutationTypePointer = 1
	mutationTypeString  = 2
	mutationTypeUint16  = 5
	mutationTypeUint32  = 6
	mutationTypeMap     = 7
	mutationTypeArray   = 11
)

// mutationBuilder assembles [tree][separator][data][marker][metadata] from
// semantic fields. Mutations change fields before build() so each fixture
// mutates exactly the field it claims to.
type mutationBuilder struct {
	ipVersion      uint
	recordSize     uint
	metaNodeCount  *uint
	metaRecordSize *uint
	nodes          [][2]uint
	separator      [dataSectionSeparatorSize]byte
	data           []byte
	marker         []byte
}

func (b *mutationBuilder) effectiveNodeCount() uint {
	if b.metaNodeCount != nil {
		return *b.metaNodeCount
	}
	return uint(len(b.nodes))
}

func (b *mutationBuilder) effectiveRecordSize() uint {
	if b.metaRecordSize != nil {
		return *b.metaRecordSize
	}
	return b.recordSize
}

// ptr encodes a data-section pointer as written by a producer for the
// actually encoded tree (len(b.nodes)), independent of any metadata override.
func (b *mutationBuilder) ptr(offset uint) uint {
	return uint(len(b.nodes)) + dataSectionSeparatorSize + offset
}

func (b *mutationBuilder) build() []byte {
	var out []byte
	for _, node := range b.nodes {
		out = appendMutationNode(out, node, b.recordSize)
	}
	out = append(out, b.separator[:]...)
	out = append(out, b.data...)
	marker := b.marker
	if marker == nil {
		marker = metadataStartMarker
	}
	out = append(out, marker...)
	out = append(out, b.metadataBytes()...)
	return out
}

func appendMutationNode(out []byte, node [2]uint, recordSize uint) []byte {
	left, right := node[0], node[1]
	switch recordSize {
	case 24:
		out = append(out, byte(left>>16), byte(left>>8), byte(left))
		out = append(out, byte(right>>16), byte(right>>8), byte(right))
	case 28:
		out = append(out,
			byte(left>>16), byte(left>>8), byte(left),
			byte((left>>24)<<4)|byte(right>>24),
			byte(right>>16), byte(right>>8), byte(right),
		)
	case 32:
		out = append(out, byte(left>>24), byte(left>>16), byte(left>>8), byte(left))
		out = append(out, byte(right>>24), byte(right>>16), byte(right>>8), byte(right))
	default:
		out = append(out, make([]byte, 2*(recordSize/4))...)
	}
	return out
}

func (b *mutationBuilder) metadataBytes() []byte {
	var out []byte
	out = append(out, encodeMutationCtrl(mutationTypeMap, 9)...)
	add := func(key string, value []byte) {
		out = append(out, encodeMutationString(key)...)
		out = append(out, value...)
	}
	add("binary_format_major_version", encodeMutationUint16(2))
	add("binary_format_minor_version", encodeMutationUint16(0))
	add("build_epoch", encodeMutationUint32(1_700_000_000))
	add("database_type", encodeMutationString("Test-Differential"))

	description := encodeMutationCtrl(mutationTypeMap, 1)
	description = append(description, encodeMutationString("en")...)
	description = append(description, encodeMutationString("Differential mutation test database")...)
	add("description", description)

	add("ip_version", encodeMutationUint16(b.ipVersion))

	languages := encodeMutationCtrl(mutationTypeArray, 1)
	languages = append(languages, encodeMutationString("en")...)
	add("languages", languages)

	add("node_count", encodeMutationUint32(b.effectiveNodeCount()))
	add("record_size", encodeMutationUint16(b.effectiveRecordSize()))
	return out
}

// encodeMutationCtrl encodes a type/size header, including the 29/30/31
// extended-size forms needed for the huge-length mutations.
func encodeMutationCtrl(typeNum, size uint) []byte {
	var sizeField uint
	var sizeBytes []byte
	switch {
	case size < 29:
		sizeField = size
	case size < 285:
		sizeField = 29
		sizeBytes = []byte{byte(size - 29)}
	case size < 65_821:
		sizeField = 30
		sizeBytes = []byte{byte((size - 285) >> 8), byte(size - 285)}
	default:
		sizeField = 31
		remaining := size - 65_821
		sizeBytes = []byte{byte(remaining >> 16), byte(remaining >> 8), byte(remaining)}
	}
	var out []byte
	if typeNum < 8 {
		out = append(out, byte(typeNum<<5|sizeField))
	} else {
		out = append(out, byte(sizeField), byte(typeNum-7))
	}
	return append(out, sizeBytes...)
}

func encodeMutationString(value string) []byte {
	out := encodeMutationCtrl(mutationTypeString, uint(len(value)))
	return append(out, value...)
}

func encodeMutationUint16(value uint) []byte {
	switch {
	case value == 0:
		return encodeMutationCtrl(mutationTypeUint16, 0)
	case value < 1<<8:
		return append(encodeMutationCtrl(mutationTypeUint16, 1), byte(value))
	default:
		return append(encodeMutationCtrl(mutationTypeUint16, 2), byte(value>>8), byte(value))
	}
}

func encodeMutationUint32(value uint) []byte {
	switch {
	case value == 0:
		return encodeMutationCtrl(mutationTypeUint32, 0)
	case value < 1<<8:
		return append(encodeMutationCtrl(mutationTypeUint32, 1), byte(value))
	case value < 1<<16:
		return append(encodeMutationCtrl(mutationTypeUint32, 2), byte(value>>8), byte(value))
	case value < 1<<24:
		return append(encodeMutationCtrl(mutationTypeUint32, 3),
			byte(value>>16), byte(value>>8), byte(value))
	default:
		return append(encodeMutationCtrl(mutationTypeUint32, 4),
			byte(value>>24), byte(value>>16), byte(value>>8), byte(value))
	}
}

// encodeMutationPointer encodes the one-byte-width pointer form, valid for
// data-section offsets below 2048 (all fixtures in this file).
func encodeMutationPointer(offset uint) []byte {
	return []byte{byte(mutationTypePointer<<5) | byte(offset>>8), byte(offset)}
}

// mutationRecordA is {"name": "alpha", "tags": ["a", "b"]}.
func mutationRecordA() []byte {
	var out []byte
	out = append(out, encodeMutationCtrl(mutationTypeMap, 2)...)
	out = append(out, encodeMutationString("name")...)
	out = append(out, encodeMutationString("alpha")...)
	out = append(out, encodeMutationString("tags")...)
	out = append(out, encodeMutationCtrl(mutationTypeArray, 2)...)
	out = append(out, encodeMutationString("a")...)
	out = append(out, encodeMutationString("b")...)
	return out
}

// mutationRecordB is {"name": "beta"}.
func mutationRecordB() []byte {
	var out []byte
	out = append(out, encodeMutationCtrl(mutationTypeMap, 1)...)
	out = append(out, encodeMutationString("name")...)
	out = append(out, encodeMutationString("beta")...)
	return out
}

// legalMutationBuilder builds a legal IPv4 database with two nodes:
//
//	node 0: left -> node 1, right -> record A   (128.0.0.0/1)
//	node 1: left -> record B, right -> empty    (0.0.0.0/2, 64.0.0.0/2)
func legalMutationBuilder() *mutationBuilder {
	recordA := mutationRecordA()
	recordB := mutationRecordB()
	data := make([]byte, 0, len(recordA)+len(recordB))
	data = append(data, recordA...)
	data = append(data, recordB...)

	b := &mutationBuilder{
		ipVersion:  4,
		recordSize: 24,
		data:       data,
	}
	b.nodes = make([][2]uint, 2)
	b.nodes[0] = [2]uint{1, b.ptr(0)}
	b.nodes[1] = [2]uint{b.ptr(uint(len(recordA))), 2}
	return b
}

func mutationConcat(parts ...[]byte) []byte {
	var out []byte
	for _, part := range parts {
		out = append(out, part...)
	}
	return out
}

// ---------------------------------------------------------------------------
// Structured mutations
// ---------------------------------------------------------------------------

type mutationExpectation int

const (
	// Every stage succeeds with no structural errors.
	expectClean mutationExpectation = iota
	// Open itself rejects the database.
	expectOpenError
	// Open succeeds but at least one later stage detects the corruption.
	expectCorrupt
	// Open and all lazy query paths succeed; only the stricter verifier
	// rejects the database.
	expectLazy
	// Every stage succeeds, but Decode and DecodePath may legitimately
	// disagree (duplicate map keys: last-wins versus first-match).
	expectDivergent
)

type structuredMutation struct {
	name   string
	expect mutationExpectation
	mutate func(*mutationBuilder)
	check  func(t *testing.T, outcome mutationOutcome)
}

func (m structuredMutation) apply() []byte {
	b := legalMutationBuilder()
	if m.mutate != nil {
		m.mutate(b)
	}
	return b.build()
}

func structuredMutations() []structuredMutation {
	// The maximum value a 31-form container header can encode.
	hugeContainerSize := uint(65_821 + (1<<24 - 1))

	return []structuredMutation{
		{
			name:   "legal-baseline",
			expect: expectClean,
		},

		// Metadata field semantics.

		{
			name:   "node-count-inflated",
			expect: expectOpenError,
			mutate: func(b *mutationBuilder) {
				nodeCount := uint(10)
				b.metaNodeCount = &nodeCount
			},
		},
		{
			name:   "node-count-shrunk",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				nodeCount := uint(1)
				b.metaNodeCount = &nodeCount
			},
		},
		{
			name:   "record-size-unsupported",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				recordSize := uint(16)
				b.metaRecordSize = &recordSize
			},
		},
		{
			name:   "record-size-mismatch-32",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				recordSize := uint(32)
				b.metaRecordSize = &recordSize
			},
		},
		{
			name:   "metadata-marker-corrupted",
			expect: expectOpenError,
			mutate: func(b *mutationBuilder) {
				marker := append([]byte(nil), metadataStartMarker...)
				marker[len(marker)-1] = 'X'
				b.marker = marker
			},
		},

		// Search-tree pointers.

		{
			name:   "tree-pointer-cycle",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				b.nodes[1] = [2]uint{1, 1}
			},
		},
		{
			// All pointers reference record A. Lookups, decodes, and
			// iteration stay consistent, but the strict verifier rejects
			// the unreferenced record B left in the data section.
			name:   "tree-pointer-fanout",
			expect: expectLazy,
			mutate: func(b *mutationBuilder) {
				shared := b.ptr(0)
				b.nodes[0] = [2]uint{shared, shared}
				b.nodes[1] = [2]uint{shared, shared}
			},
		},
		{
			name:   "tree-pointer-into-separator",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				b.nodes[1][0] = uint(len(b.nodes)) + 8
			},
		},
		{
			name:   "tree-pointer-beyond-data",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				b.nodes[1][0] = b.ptr(uint(len(b.data))) + 50
			},
		},

		// Data-section separator.

		{
			name:   "separator-nonzero",
			expect: expectLazy,
			mutate: func(b *mutationBuilder) {
				b.separator[7] = 0xAB
			},
		},

		// Container length headers.

		{
			name:   "map-length-overrun",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				// Record A claims three entries but encodes two.
				b.data[0] = 0xE0 | 3
			},
		},
		{
			name:   "map-length-huge",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				huge := encodeMutationCtrl(mutationTypeMap, hugeContainerSize)
				recordB := mutationRecordB()
				b.data = mutationConcat(huge, recordB)
				b.nodes[1][0] = b.ptr(uint(len(huge)))
			},
		},
		{
			name:   "array-length-huge",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				huge := encodeMutationCtrl(mutationTypeArray, hugeContainerSize)
				recordB := mutationRecordB()
				b.data = mutationConcat(huge, recordB)
				b.nodes[1][0] = b.ptr(uint(len(huge)))
			},
		},

		// Truncated values at the end of the data section.

		{
			name:   "truncated-varint-uint32",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				recordA := mutationRecordA()
				truncated := encodeMutationCtrl(mutationTypeMap, 1)
				truncated = append(truncated, encodeMutationString("epoch")...)
				// uint32 header claims four payload bytes; two are present.
				truncated = append(truncated, 0xC4, 0x01, 0x02)
				b.data = mutationConcat(recordA, truncated)
				b.nodes[1][0] = b.ptr(uint(len(recordA)))
			},
		},
		{
			name:   "truncated-string-length",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				recordA := mutationRecordA()
				// String header with the two-byte length form, but only
				// one length byte is present before the data section ends.
				truncated := []byte{0x40 | 30, 0x01}
				b.data = mutationConcat(recordA, truncated)
				b.nodes[1][0] = b.ptr(uint(len(recordA)))
			},
		},

		// Shared data-section pointers.

		{
			name:   "data-pointer-cycle",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				recordA := mutationRecordA()
				cycleOffset := uint(len(recordA))
				b.data = mutationConcat(recordA, encodeMutationPointer(cycleOffset))
				b.nodes[1][0] = b.ptr(cycleOffset)
			},
		},
		{
			// Record A shares one string through two pointers. The shared
			// string is also reachable from the tree so the strict
			// verifier accepts the database.
			name:   "data-pointer-fanout-shared",
			expect: expectClean,
			mutate: func(b *mutationBuilder) {
				header := encodeMutationCtrl(mutationTypeMap, 2)
				nameKey := encodeMutationString("name")
				aliasKey := encodeMutationString("alias")
				sharedOffset := uint(len(header) + len(nameKey) + 2 + len(aliasKey) + 2)

				var recordA []byte
				recordA = append(recordA, header...)
				recordA = append(recordA, nameKey...)
				recordA = append(recordA, encodeMutationPointer(sharedOffset)...)
				recordA = append(recordA, aliasKey...)
				recordA = append(recordA, encodeMutationPointer(sharedOffset)...)

				shared := encodeMutationString("shared")
				recordB := mutationRecordB()
				recordBOffset := sharedOffset + uint(len(shared))

				b.data = mutationConcat(recordA, shared, recordB)
				b.nodes[1][0] = b.ptr(recordBOffset)
				b.nodes[1][1] = b.ptr(sharedOffset)
			},
		},

		// Payload validity.

		{
			// Lazy decoding delivers the invalid UTF-8 bytes as a Go
			// string; only the verifier rejects them.
			name:   "utf8-invalid-payload",
			expect: expectLazy,
			mutate: func(b *mutationBuilder) {
				var recordA []byte
				recordA = append(recordA, encodeMutationCtrl(mutationTypeMap, 1)...)
				recordA = append(recordA, encodeMutationString("name")...)
				recordA = append(recordA, 0x43, 0xFF, 0xFE, 0xFD)
				recordB := mutationRecordB()
				b.data = mutationConcat(recordA, recordB)
				b.nodes[1][0] = b.ptr(uint(len(recordA)))
			},
		},
		{
			// Duplicate map keys: Decode into a Go map keeps the last
			// entry, DecodePath returns the first match, and the verifier
			// currently accepts the database. Both values stay within the
			// declared data section and decode budget.
			name:   "duplicate-map-key",
			expect: expectDivergent,
			mutate: func(b *mutationBuilder) {
				var recordA []byte
				recordA = append(recordA, encodeMutationCtrl(mutationTypeMap, 3)...)
				recordA = append(recordA, encodeMutationString("name")...)
				recordA = append(recordA, encodeMutationString("first")...)
				recordA = append(recordA, encodeMutationString("name")...)
				recordA = append(recordA, encodeMutationString("second")...)
				recordA = append(recordA, encodeMutationString("tags")...)
				recordA = append(recordA, encodeMutationCtrl(mutationTypeArray, 1)...)
				recordA = append(recordA, encodeMutationString("a")...)
				recordB := mutationRecordB()
				b.data = mutationConcat(recordA, recordB)
				b.nodes[1][0] = b.ptr(uint(len(recordA)))
			},
			check: func(t *testing.T, outcome mutationOutcome) {
				t.Helper()
				require.NoError(t, outcome.verifyErr,
					"verifier currently accepts duplicate map keys")
				require.True(t, outcome.valueDivergence,
					"expected Decode (last-wins) and DecodePath (first-match) to disagree")
			},
		},

		// IPv4 start node corruption in an IPv6 database.

		{
			name:   "ipv4-start-subtree-cycle",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				b.ipVersion = 6
				b.data = nil
				b.nodes = [][2]uint{{1, 2}, {1, 1}}
			},
		},
		{
			name:   "ipv4-start-pointer-into-separator",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				b.ipVersion = 6
				b.data = nil
				b.nodes = [][2]uint{{1, 2}, {10, 2}}
			},
		},

		// Errors surfacing in the middle of network iteration.

		{
			name:   "networks-mid-iteration-error",
			expect: expectCorrupt,
			mutate: func(b *mutationBuilder) {
				// 64.0.0.0/2 points into the separator region; iteration
				// yields record B first and fails afterwards.
				b.nodes[1][1] = uint(len(b.nodes)) + 8
			},
			check: func(t *testing.T, outcome mutationOutcome) {
				t.Helper()
				require.Positive(t, outcome.networksYielded,
					"expected at least one network before the mid-iteration error")
				require.Error(t, outcome.networksErr,
					"expected a mid-iteration error after %d networks", outcome.networksYielded)
			},
		},
	}
}

// ---------------------------------------------------------------------------
// Differential harness
// ---------------------------------------------------------------------------

// mutationOutcome records what each stage did with the same fixture. Error
// fields are mutually independent: stages are allowed to fail at different
// times, and the tests only reject contradictory *successes*.
type mutationOutcome struct {
	openErr           error
	verifyErr         error
	lookupErr         error
	decodeErr         error
	decodePathErr     error
	networksErr       error
	networksDecodeErr error
	networksYielded   int
	// valueDivergence records a case where full Decode and DecodePath both
	// succeeded but selected different values (duplicate map keys). It is
	// permitted for corrupt databases; strict-equivalence fixtures fail on it.
	valueDivergence bool
}

func (o mutationOutcome) laterStageFailed() bool {
	return o.verifyErr != nil ||
		o.lookupErr != nil ||
		o.decodeErr != nil ||
		o.decodePathErr != nil ||
		o.networksErr != nil ||
		o.networksDecodeErr != nil
}

// detected is only consulted when openErr is nil.
func (o mutationOutcome) detected() bool {
	return o.laterStageFailed()
}

func isBenignMutationLookupError(err error) bool {
	// Looking up an IPv6 address in an IPv4-only database is a caller
	// argument error, not database corruption.
	return err != nil && strings.Contains(err.Error(), "IPv4-only database")
}

func openMutationViaTempFile(t *testing.T, data []byte) error {
	t.Helper()
	path := filepath.Join(t.TempDir(), "mutation.mmdb")
	require.NoError(t, os.WriteFile(path, data, 0o600))
	reader, err := Open(path)
	if err != nil {
		return err
	}
	return reader.Close()
}

// exerciseMutation runs every read path against one fixture and enforces the
// cross-stage invariants. maxNetworks bounds iteration; crossCheckOpen adds a
// temp-file Open (disabled during fuzzing).
func exerciseMutation(
	t *testing.T,
	name string,
	data []byte,
	maxNetworks int,
	crossCheckOpen bool,
	strictEquivalence bool,
) mutationOutcome {
	t.Helper()
	var outcome mutationOutcome

	reader, err := OpenBytes(data)
	outcome.openErr = err

	if crossCheckOpen {
		tempErr := openMutationViaTempFile(t, data)
		require.Equal(t, outcome.openErr == nil, tempErr == nil,
			"mutation %s: OpenBytes error %v but Open error %v",
			name, outcome.openErr, tempErr)
	}

	if outcome.openErr != nil {
		return outcome
	}
	t.Cleanup(func() { require.NoError(t, reader.Close()) })

	outcome.verifyErr = reader.Verify()

	lookupIPs := []string{"200.0.0.0", "10.0.0.0", "::"}
	for _, ipString := range lookupIPs {
		ip := netip.MustParseAddr(ipString)
		result := reader.Lookup(ip)
		lookupErr := result.Err()
		if isBenignMutationLookupError(lookupErr) {
			continue
		}
		if lookupErr != nil {
			if outcome.lookupErr == nil {
				outcome.lookupErr = lookupErr
			}
			continue
		}
		if !result.Found() {
			continue
		}

		// No successful lookup may point outside the declared data section.
		require.Less(t, result.Offset(), uintptr(reader.dataSectionSize),
			"mutation %s: lookup %s returned offset %d outside the declared data section (%d bytes)",
			name, ipString, result.Offset(), reader.dataSectionSize)

		var decoded any
		decodeErr := result.Decode(&decoded)
		if decodeErr != nil {
			if outcome.decodeErr == nil {
				outcome.decodeErr = decodeErr
			}
		} else {
			assertMutationBudget(t, name, "Decode "+ipString, decoded)
		}

		var leaf any
		pathErr := result.DecodePath(&leaf, "name")
		if pathErr != nil {
			if outcome.decodePathErr == nil {
				outcome.decodePathErr = pathErr
			}
		} else {
			assertMutationBudget(t, name, "DecodePath "+ipString, leaf)
		}

		// Equivalence property: both paths may fail at different times,
		// but they must never disagree about a successfully decoded value
		// for a database known to be legal. For corrupt databases (e.g.
		// duplicate map keys) the divergence is recorded, not rejected;
		// every individual success was already budget- and bounds-checked.
		if decodeErr == nil {
			walked, ok := walkMutationPath(decoded, "name")
			switch {
			case ok:
				if pathErr != nil {
					if strictEquivalence {
						require.NoError(t, pathErr,
							"mutation %s: Decode reached path \"name\" but DecodePath failed", name)
					}
				} else if !reflect.DeepEqual(walked, leaf) {
					if strictEquivalence {
						require.Equal(t, walked, leaf,
							"mutation %s: DecodePath leaf disagrees with full Decode at %s",
							name, ipString)
					}
					outcome.valueDivergence = true
				}
			case pathErr == nil:
				if leaf != nil {
					if strictEquivalence {
						require.Nil(t, leaf,
							"mutation %s: DecodePath returned %#v for path absent from Decode",
							name, leaf)
					}
					outcome.valueDivergence = true
				}
			}
		}
	}

	outcome.networksYielded,
		outcome.networksErr,
		outcome.networksDecodeErr = exerciseMutationNetworks(t, name, reader, maxNetworks)

	// A successful verification is a strong statement: no later stage may
	// report a structural error on data decoded into any.
	if outcome.verifyErr == nil {
		require.NoError(t, outcome.lookupErr,
			"mutation %s: Verify passed but Lookup failed", name)
		require.NoError(t, outcome.decodeErr,
			"mutation %s: Verify passed but Decode failed", name)
		require.NoError(t, outcome.networksErr,
			"mutation %s: Verify passed but Networks failed", name)
		require.NoError(t, outcome.networksDecodeErr,
			"mutation %s: Verify passed but a Networks decode failed", name)
	}

	// Record (rather than reject) permitted lazy-open/later-failure cases.
	if outcome.laterStageFailed() {
		t.Logf(
			"mutation %s: lazy open with later failure: verify=%v lookup=%v decode=%v decodePath=%v networks=%v networksDecode=%v",
			name,
			outcome.verifyErr,
			outcome.lookupErr,
			outcome.decodeErr,
			outcome.decodePathErr,
			outcome.networksErr,
			outcome.networksDecodeErr,
		)
	}

	return outcome
}

func exerciseMutationNetworks(
	t *testing.T,
	name string,
	reader *Reader,
	maxResults int,
) (int, error, error) {
	t.Helper()
	yielded := 0
	for result := range reader.Networks() {
		if err := result.Err(); err != nil {
			return yielded, err, nil
		}
		if !result.Found() {
			continue
		}
		yielded++

		require.Less(t, result.Offset(), uintptr(reader.dataSectionSize),
			"mutation %s: Networks returned offset %d outside the declared data section (%d bytes)",
			name, result.Offset(), reader.dataSectionSize)

		var decoded any
		if err := result.Decode(&decoded); err != nil {
			return yielded, nil, err
		}
		assertMutationBudget(t, name, "Networks.Decode", decoded)

		if yielded >= maxResults {
			return yielded, nil, nil
		}
	}
	return yielded, nil, nil
}

// assertMutationBudget measures a successfully decoded value in the same
// units as the decoder's documented operation limits.
func assertMutationBudget(t *testing.T, name, stage string, value any) {
	t.Helper()
	slots, payload := measureMutationValue(value, 0)
	require.LessOrEqual(t, slots, mutationMaxContainerSlots,
		"mutation %s stage %s: decoded value uses %d container slots, budget is %d",
		name, stage, slots, mutationMaxContainerSlots)
	require.LessOrEqual(t, payload, mutationMaxPayloadBytes,
		"mutation %s stage %s: decoded value uses %d payload bytes, budget is %d",
		name, stage, payload, mutationMaxPayloadBytes)
}

// walkMutationPath follows map keys and array indices (including negative
// indices) through a value produced by Decode into any.
func walkMutationPath(value any, path ...any) (any, bool) {
	current := value
	for _, segment := range path {
		switch key := segment.(type) {
		case string:
			m, ok := current.(map[string]any)
			if !ok {
				return nil, false
			}
			current, ok = m[key]
			if !ok {
				return nil, false
			}
		case int:
			s, ok := current.([]any)
			if !ok {
				return nil, false
			}
			index := key
			if index < 0 {
				index += len(s)
			}
			if index < 0 || index >= len(s) {
				return nil, false
			}
			current = s[index]
		default:
			return nil, false
		}
	}
	return current, true
}

// measureMutationValue counts container child slots and materialized
// payload bytes of a decoded value.
func measureMutationValue(value any, depth int) (slots, payload int) {
	if depth > mutationMaxMeasureDepth {
		return 0, 0
	}
	switch typed := value.(type) {
	case map[string]any:
		slots += 2 * len(typed)
		for key, item := range typed {
			payload += len(key)
			childSlots, childPayload := measureMutationValue(item, depth+1)
			slots += childSlots
			payload += childPayload
		}
	case []any:
		slots += len(typed)
		for _, item := range typed {
			childSlots, childPayload := measureMutationValue(item, depth+1)
			slots += childSlots
			payload += childPayload
		}
	case string:
		payload += len(typed)
	case []byte:
		payload += len(typed)
	}
	return slots, payload
}

// ---------------------------------------------------------------------------
// Tests and directed fuzz seeds
// ---------------------------------------------------------------------------

func TestStructuredMutations(t *testing.T) {
	for _, mutation := range structuredMutations() {
		t.Run(mutation.name, func(t *testing.T) {
			data := mutation.apply()
			outcome := exerciseMutation(
				t, mutation.name, data, mutationMaxNetworks, true,
				mutation.expect == expectClean,
			)

			switch mutation.expect {
			case expectClean:
				require.NoError(t, outcome.openErr,
					"mutation %s: expected open to succeed", mutation.name)
				require.False(t, outcome.laterStageFailed(),
					"mutation %s: expected every stage to succeed, got %+v",
					mutation.name, outcome)
			case expectOpenError:
				require.Error(t, outcome.openErr,
					"mutation %s: expected open to fail", mutation.name)
			case expectCorrupt:
				if outcome.openErr == nil {
					require.True(t, outcome.detected(),
						"mutation %s: corruption escaped every stage, got %+v",
						mutation.name, outcome)
				}
			case expectLazy:
				require.NoError(t, outcome.openErr,
					"mutation %s: expected lazy open to succeed", mutation.name)
				require.Error(t, outcome.verifyErr,
					"mutation %s: expected Verify to reject the fixture", mutation.name)
				require.NoError(t, outcome.lookupErr,
					"mutation %s: lazy Lookup must still work", mutation.name)
				require.NoError(t, outcome.decodeErr,
					"mutation %s: lazy Decode must still work", mutation.name)
				require.NoError(t, outcome.networksErr,
					"mutation %s: lazy Networks must still work", mutation.name)
				require.NoError(t, outcome.networksDecodeErr,
					"mutation %s: lazy Networks decoding must still work", mutation.name)
			case expectDivergent:
				require.NoError(t, outcome.openErr,
					"mutation %s: expected open to succeed", mutation.name)
				require.False(t, outcome.laterStageFailed(),
					"mutation %s: expected every stage to succeed, got %+v",
					mutation.name, outcome)
			}

			if mutation.check != nil {
				mutation.check(t, outcome)
			}
		})
	}
}

// TestDecodePathEquivalenceOnLegalDatabase checks the equivalence property on
// every network of the legal fixture, including nested, negative, and
// missing indices.
func TestDecodePathEquivalenceOnLegalDatabase(t *testing.T) {
	reader, err := OpenBytes(legalMutationBuilder().build())
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, reader.Close()) })
	require.NoError(t, reader.Verify())

	paths := [][]any{
		{"name"},
		{"tags", 0},
		{"tags", 1},
		{"tags", -1},
		{"tags", 99},
		{"missing"},
		{"missing", "path"},
	}

	checked := 0
	for result := range reader.Networks() {
		require.NoError(t, result.Err())
		require.True(t, result.Found())

		var whole any
		require.NoError(t, result.Decode(&whole))
		assertMutationBudget(t, "legal-baseline", "Decode", whole)

		for _, path := range paths {
			var leaf any
			require.NoError(t, result.DecodePath(&leaf, path...),
				"path %v", path)
			assertMutationBudget(t, "legal-baseline", "DecodePath", leaf)

			walked, ok := walkMutationPath(whole, path...)
			if ok {
				require.Equal(t, walked, leaf,
					"path %v: DecodePath disagrees with full Decode", path)
			} else {
				require.Nil(t, leaf,
					"path %v: DecodePath returned %#v for an absent path", path, leaf)
			}
		}
		checked++
	}
	require.Positive(t, checked, "expected at least one network in the legal fixture")
}

// FuzzStructuredMutation runs the same cross-stage invariants against fuzz
// input. The seeds are directed fixtures produced by the mutation builder.
func FuzzStructuredMutation(f *testing.F) {
	seeds := map[string]bool{
		"legal-baseline":           true,
		"tree-pointer-cycle":       true,
		"map-length-huge":          true,
		"truncated-varint-uint32":  true,
		"separator-nonzero":        true,
		"ipv4-start-subtree-cycle": true,
		"data-pointer-cycle":       true,
		"duplicate-map-key":        true,
	}
	for _, mutation := range structuredMutations() {
		if seeds[mutation.name] {
			f.Add(mutation.apply())
		}
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 || len(data) > mutationMaxFuzzSize {
			return
		}
		// Fuzzed databases may be corrupt, so value-level equivalence is
		// not asserted here; bounds and budget invariants always are.
		exerciseMutation(t, "fuzz", data, mutationMaxFuzzNetworks, false, false)
	})
}
