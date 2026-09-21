package maxminddb

// Differential mutation cases, harness, and equivalence-path properties.
// See mutation_diff_test.go for the fixture/builder semantics.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"net/netip"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/oschwald/maxminddb-golang/v2/internal/mmdberrors"
)

// Probe addresses for the IPv4 fixture. 10.0.0.1 lands on leafA (00 prefix),
// 64.0.0.1 lands on leafB (01 prefix), 128.0.0.1 lands on an empty record.
var (
	probeLeafA   = netip.MustParseAddr("10.0.0.1")
	probeLeafB   = netip.MustParseAddr("64.0.0.1")
	probeMissing = netip.MustParseAddr("128.0.0.1")
)

// Explicit caps keep crafted fixtures from turning into timeouts or memory
// blowups. They are deliberately far above the constant-size fixtures but
// small enough to fail loudly if a bound regresses.
const (
	networksIterationCap = 4096
	maxMaterializedBytes = 2 << 20 // mirrors decoder.decodePayloadBudgetBytes
	maxContainerSlots    = 32_768  // mirrors the decoder child-slot allowance
)

type mutationCase struct {
	name      string
	fixture   func() *fixtureSpec
	openErr   string // "" means OpenBytes is expected to succeed
	verifyErr string // "" means Verify is expected to pass
	// leaf expectations keyed by probe address; nil ip => lookup itself errors.
	leafAErr string
	leafBErr string
	// expectLazyOpen records the deliberate boundary case: Open succeeds
	// while Verify rejects the same bytes.
	skipProbes bool
}

func catalog() []mutationCase {
	base := func() *fixtureSpec { return newMinimalFixture(24) }

	cases := []mutationCase{
		{name: "node-count-overcount", fixture: func() *fixtureSpec {
			f := base()
			f.setMeta("node_count", metaU32(8))
			return f
		}, verifyErr: "search tree is corrupt",
			leafAErr: "search tree is corrupt", leafBErr: "search tree is corrupt"},
		{name: "node-count-undercount", fixture: func() *fixtureSpec {
			f := base()
			f.setMeta("node_count", metaU32(2))
			return f
		}, verifyErr: "data separator", skipProbes: true},
		{name: "node-count-huge-overflow", fixture: func() *fixtureSpec {
			f := base()
			f.setMeta("node_count", metaU64(1<<40))
			return f
		}, openErr: "invalid metadata"},
		{name: "record-size-28-mismatch", fixture: func() *fixtureSpec {
			f := base()
			f.setMeta("record_size", metaU16(28))
			return f
		}, verifyErr: "search tree",
			leafAErr: "search tree", leafBErr: "search tree"},
		{name: "record-size-invalid-33", fixture: func() *fixtureSpec {
			f := base()
			f.setMeta("record_size", metaU16(33))
			return f
		}, verifyErr: "24, 28, or 32",
			leafAErr: "unsupported record size", leafBErr: "unsupported record size"},
		{name: "ip-version-invalid-5", fixture: func() *fixtureSpec {
			f := base()
			f.setMeta("ip_version", metaU16(5))
			return f
		}, verifyErr: "4 or 6"},
		{name: "node-count-varint-truncated", fixture: func() *fixtureSpec {
			f := base()
			// node_count is the last field: declare a uint32 of size 3 then
			// cut the final payload byte off the end of the file.
			f.setMeta("node_count", []byte{0x83, 0x00, 0x00, 0x00})
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				db.buf = db.buf[:len(db.buf)-1]
			}
			return f
		}, openErr: "unexpected end of database"},
		{name: "metadata-marker-corrupt", fixture: func() *fixtureSpec {
			f := base()
			f.markerOK = false
			return f
		}, openErr: "invalid MaxMind DB file"},
		{name: "separator-nonzero", fixture: func() *fixtureSpec {
			f := base()
			f.separator[7] = 0xFF
			return f
		}, verifyErr: "unexpected byte in data separator"},
		{name: "tree-record-oob-pointer", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				bad := db.nodeCount + dataSectionSeparatorSize + db.dataLen + 100
				// Corrupt the record on the leafA path (node 1, left).
				patchLeftRecord(db, 1, bad)
			}
			return f
		}, verifyErr: "search tree is corrupt",
			leafAErr: "search tree is corrupt"},
		{name: "tree-record-into-separator", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				patchLeftRecord(db, 1, db.nodeCount+1)
			}
			return f
		}, verifyErr: "search tree is corrupt",
			leafAErr: "search tree is corrupt"},
		{name: "tree-cycle-at-root", fixture: func() *fixtureSpec {
			f := base()
			// Rebuild the tree so the zero-bit path self-loops: node 0's
			// left record points back at node 0. The leafA IP starts with a
			// zero bit, so Lookup must reject rather than return data.
			self := nodeRefTo(0)
			f.nodes = [][2]nodeRef{
				{self, self},
				{emptyRef(), emptyRef()},
				{emptyRef(), emptyRef()},
				{dataRef("leafA"), dataRef("leafB")},
			}
			return f
		}, verifyErr: "cycle",
			leafAErr: "invalid node in search tree",
			leafBErr: "invalid node in search tree"},
		{name: "tree-deep-path-128", fixture: func() *fixtureSpec {
			f := base()
			// 130 internal nodes in a straight chain on the zero-bit path
			// exceed the 128-bit maximum. The leafA IP begins with zero
			// bits, so lookup walks the chain and must reject the tree.
			const count = 130
			nodes := make([][2]nodeRef, count)
			for i := range nodes {
				nodes[i] = [2]nodeRef{emptyRef(), emptyRef()}
			}
			for i := uint(0); i+1 < count; i++ {
				nodes[i][0] = nodeRefTo(i + 1)
			}
			nodes[count-1][0] = dataRef("leafA")
			f.nodes = nodes
			f.setMeta("node_count", metaU32(count))
			return f
		}, verifyErr: "128", skipProbes: true},
		{name: "data-map-huge-length", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				// leafA root map (offset 0): widen declared size to 40,000
				// with a 2-byte extended length header.
				off := db.dataStart
				v := uint(40_000 - 285)
				db.buf[off] = 7<<5 | 30
				db.buf[off+1] = byte(v >> 8)
				db.buf[off+2] = byte(v)
			}
			return f
		}, verifyErr: "decoding error",
			leafAErr: "maximum decoded record size"},
		{name: "data-array-huge-length", fixture: func() *fixtureSpec {
			f := base()
			// Widen leafA's tags array to size 40,000 via a 2-byte header.
			chunk := f.chunks[0]
			// Extended slice header: low-five size code 30, extended kind
			// byte (KindSlice-7 = 4), then the two-byte extended length.
			wide := uint(40_000 - 285)
			chunk.replaceAtLabel("tagsArr", 1,
				[]byte{0x1E, 0x04, byte(wide >> 8), byte(wide)})
			return f
		}, verifyErr: "decoding error",
			leafAErr: "maximum decoded record size"},
		{name: "data-string-length-runs-past-end", fixture: func() *fixtureSpec {
			f := base()
			// leafB "ja": declare 200 bytes via size code 29 (+29), which
			// overshoots the chunk without changing its byte length.
			chunk := f.chunks[1]
			chunk.replaceAtLabel("bName", 1, []byte{2<<5 | 29, 200 - 29})
			return f
		}, verifyErr: "decoding error",
			leafBErr: "unexpected end of database"},
		{name: "data-pointer-to-pointer", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				at := db.dataStart + db.labelOffset["namePtr:ptr"]
				encodePointer(db.buf, at, db.labelOffset["namePtr:ptr"])
			}
			return f
		}, verifyErr: "decoding error",
			leafAErr: "invalid pointer to pointer"},
		{name: "data-pointer-beyond-section", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				at := db.dataStart + db.labelOffset["namePtr:ptr"]
				db.buf[at] = 0x38
				binary.BigEndian.PutUint32(db.buf[at+1:at+5],
					uint32(db.dataLen+5000))
			}
			return f
		}, verifyErr: "decoding error",
			leafAErr: "unexpected end of database"},
		{name: "data-invalid-utf8-shared-string", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				off := db.dataStart + db.labelOffset["shared"]
				db.buf[off+1] = 0xFF // first payload byte of "en"
			}
			return f
		}, verifyErr: "invalid UTF-8", leafAErr: ""},
		{name: "file-truncated-metadata-tail", fixture: func() *fixtureSpec {
			f := base()
			f.rawPatch = func(_ *testing.T, db *assembledDB) {
				db.buf = db.buf[:len(db.buf)-3]
			}
			return f
		}, openErr: "unexpected end of database"},
	}
	return cases
}

// patchLeftRecord rewrites the left record of a node with an exact raw value.
func patchLeftRecord(db *assembledDB, node, value uint) {
	mult := db.recordSize / 4
	base := node * mult
	switch db.recordSize {
	case 24:
		db.buf[base] = byte(value >> 16)
		db.buf[base+1] = byte(value >> 8)
		db.buf[base+2] = byte(value)
	case 28:
		db.buf[base] = byte(value >> 16)
		db.buf[base+1] = byte(value >> 8)
		db.buf[base+2] = byte(value)
		db.buf[base+3] = (db.buf[base+3] & 0x0F) | byte(value>>20)<<4
	case 32:
		binary.BigEndian.PutUint32(db.buf[base:base+4], uint32(value))
	}
}

// isInvalidDatabase reports whether err is a structural InvalidDatabaseError
// rather than a type mismatch or usage error. Successful decodes in the
// differential harness must never sit alongside a structural failure of the
// same bytes without both being individually bounded.
func isInvalidDatabase(err error) bool {
	var target mmdberrors.InvalidDatabaseError
	return errors.As(err, &target)
}

// isPathShapeMismatch reports an error from DecodePath caused by navigating a
// path against a differently-shaped value (e.g. a map key into a scalar).
// Such failures are legitimate navigation errors, not database corruption.
func isPathShapeMismatch(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "expected a map for") ||
		strings.Contains(msg, "expected a slice for")
}

// assertDecodedWithinBudget checks every successful decode against the declared
// data section and the documented payload/container budgets.
func assertDecodedWithinBudget(t *testing.T, db assembledDB, label string, value any) {
	t.Helper()
	total := materializedSize(t, value, 0)
	if total > maxMaterializedBytes {
		t.Fatalf("%s: materialized payload %d exceeds %d-byte budget",
			label, total, maxMaterializedBytes)
	}
	if str, ok := value.(string); ok {
		if len(str) > maxMaterializedBytes {
			t.Fatalf("%s: single string %d exceeds payload budget", label, len(str))
		}
	}
	// All strings come from the declared data section; a successful decode of
	// a string whose bytes are outside that window would cross the declared
	// boundary.
	for _, str := range allStrings(value) {
		if !substringOf(db.dataSection(), str) {
			t.Fatalf("%s: decoded string %q not contained in declared data section",
				label, str)
		}
	}
}

func substringOf(haystack []byte, needle string) bool {
	if len(needle) == 0 {
		return true
	}
	if len(haystack) < len(needle) {
		return false
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if string(haystack[i:i+len(needle)]) == needle {
			return true
		}
	}
	return false
}

// materializedSize recursively totals string/byte payload and counts declared
// container slots, enforcing the same allowances documented on Result.Decode.
func materializedSize(t *testing.T, v any, slots int) int {
	t.Helper()
	if slots > maxContainerSlots {
		t.Fatalf("decoded container slots exceed %d", maxContainerSlots)
	}
	switch x := v.(type) {
	case nil:
		return 0
	case string:
		return len(x)
	case []byte:
		return len(x)
	case map[string]any:
		total := 0
		for k, item := range x {
			total += len(k)
			total += materializedSize(t, item, slots+1)
			slots += 2
			if slots > maxContainerSlots {
				t.Fatalf("decoded map slots exceed %d", maxContainerSlots)
			}
		}
		return total
	case []any:
		total := 0
		for _, item := range x {
			total += materializedSize(t, item, slots+1)
		}
		return total
	default:
		return 0
	}
}

func allStrings(v any) []string {
	switch x := v.(type) {
	case string:
		return []string{x}
	case map[string]any:
		var out []string
		for k, item := range x {
			out = append(out, k)
			out = append(out, allStrings(item)...)
		}
		return out
	case []any:
		var out []string
		for _, item := range x {
			out = append(out, allStrings(item)...)
		}
		return out
	default:
		return nil
	}
}

// navigateGo mirrors Result.DecodePath navigation over the value produced by a
// full Result.Decode: string elements select map keys, ints index arrays,
// negatives count from the end. Missing paths report ok=false.
func navigateGo(v any, path []any) (any, bool) {
	cur := v
	for _, el := range path {
		switch e := el.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				return nil, false
			}
			cur, ok = m[e]
			if !ok {
				return nil, false
			}
		case int:
			arr, ok := cur.([]any)
			if !ok {
				return nil, false
			}
			idx := e
			if idx < 0 {
				idx = len(arr) + idx
			}
			if idx < 0 || idx >= len(arr) {
				return nil, false
			}
			cur = arr[idx]
		default:
			return nil, false
		}
	}
	return cur, true
}

// runLeafProbe exercises Lookup + Decode + every DecodePath probe for one
// leaf IP. On a structurally intact record it enforces the equivalence-path
// property against a full Decode. On a corrupted record it enforces that
// partial successes stay inside declared bounds while failures are structural.
func runLeafProbe(
	t *testing.T,
	reader *Reader,
	db assembledDB,
	ip netip.Addr,
	label string,
	paths []probePath,
	wantErr string,
	skipProbes bool,
) {
	t.Helper()

	result := reader.Lookup(ip)
	if result.Err() != nil {
		if wantErr == "" {
			t.Fatalf("%s: unexpected lookup error: %v", label, result.Err())
		}
		require.ErrorContains(t, result.Err(), wantErr)
		require.True(t, isInvalidDatabase(result.Err()),
			"%s: lookup error is not a structural InvalidDatabaseError: %v",
			label, result.Err())
		return
	}
	require.NoError(t, result.Err())

	var full any
	fullErr := result.Decode(&full)
	if skipProbes && wantErr == "" && fullErr != nil {
		// Corruption rerouted the address to garbage; only the structural
		// nature of the failure and the lookup bounds are meaningful here.
		require.True(t, isInvalidDatabase(fullErr),
			"%s: non-structural decode error: %v", label, fullErr)
		return
	}
	if wantErr != "" {
		// Lookup itself was fine; the failure must surface during decode.
		require.Error(t, fullErr, "%s: expected decode failure", label)
		require.ErrorContains(t, fullErr, wantErr)
		require.True(t, isInvalidDatabase(fullErr),
			"%s: decode failure is not structural: %v", label, fullErr)

		// Differential boundary: paths that still succeed must nevertheless
		// stay inside declared data section and payload budgets; paths that
		// fail must do so structurally. A path must never return data beyond
		// the declared region while the full decode reports a structure error.
		for _, p := range paths {
			var byPath any
			pathErr := result.DecodePath(&byPath, p.elems...)
			if pathErr == nil {
				assertDecodedWithinBudget(t, db, label+":"+p.name, byPath)
				continue
			}
			require.True(t, isInvalidDatabase(pathErr),
				"%s path %s: non-structural error %v", label, p.name, pathErr)
		}
		return
	}

	require.NoError(t, fullErr, "%s: expected successful full decode", label)
	assertDecodedWithinBudget(t, db, label+":full", full)
	if skipProbes {
		// The corruption reroutes this address to a different or empty
		// record, so equivalence against the original leaf does not apply.
		// The bounded decode above plus runNetworksBounded still prove no
		// successful read crosses the declared section or budgets.
		return
	}
	require.True(t, result.Found(), "%s: expected the leaf IP to be found", label)

	// An empty DecodePath is exactly a full Decode.
	var rootPath any
	require.NoError(t, result.DecodePath(&rootPath))
	require.Equal(t, full, rootPath,
		"%s: DecodePath with no path must equal Decode", label)

	for _, p := range paths {
		var byPath any
		err := result.DecodePath(&byPath, p.elems...)
		require.NoError(t, err, "%s path %s", label, p.name)
		assertDecodedWithinBudget(t, db, label+":"+p.name, byPath)

		expected, ok := navigateGo(full, p.elems)
		require.True(t, ok, "%s path %s missing in full decode", label, p.name)
		require.Equal(t, expected, byPath,
			"%s path %s: DecodePath leaf differs from full Decode value",
			label, p.name)
	}
}

// runNetworksBounded drives Networks with an explicit result cap so a corrupt
// tree can never turn iteration into a timeout or memory blowup. Any error
// observed mid-iteration must be structural; any successful decode is checked
// against the declared data section and budgets.
func runNetworksBounded(t *testing.T, reader *Reader, db assembledDB) {
	t.Helper()
	count := 0
	for iter := range reader.Networks(IncludeNetworksWithoutData()) {
		count++
		if count > networksIterationCap {
			t.Fatalf("Networks exceeded %d results on a constant-size fixture; "+
				"a corrupt cycle or fan-out is not bounded", networksIterationCap)
		}
		if iter.Err() != nil {
			require.True(t, isInvalidDatabase(iter.Err()),
				"non-structural iteration error: %v", iter.Err())
			continue
		}
		if !iter.Found() {
			continue // included empty network
		}
		var value any
		err := iter.Decode(&value)
		switch {
		case err == nil:
			assertDecodedWithinBudget(t, db, fmt.Sprintf("network:%s", iter.Prefix()), value)
		default:
			require.True(t, isInvalidDatabase(err),
				"non-structural iteration decode error: %v", err)
		}
	}
}

// TestMutationDifferential runs every semantic mutation through Open, Verify,
// Lookup + Decode/DecodePath, and a capped Networks traversal.
func TestMutationDifferential(t *testing.T) {
	for _, tc := range catalog() {
		t.Run(tc.name, func(t *testing.T) {
			f := tc.fixture()
			db := f.assemble(t)

			reader, err := OpenBytes(db.buf)
			if tc.openErr != "" {
				require.ErrorContains(t, err, tc.openErr,
					"mutation %s should be rejected while opening", tc.name)
				return
			}
			require.NoError(t, err, "mutation %s: OpenBytes failed", tc.name)
			t.Cleanup(func() { require.NoError(t, reader.Close()) })

			verifyErr := reader.Verify()
			switch {
			case tc.verifyErr != "":
				require.ErrorContains(t, verifyErr, tc.verifyErr,
					"mutation %s: Verify should reject", tc.name)
				require.True(t, isInvalidDatabase(verifyErr),
					"mutation %s: Verify error is not structural: %v",
					tc.name, verifyErr)
				// The documented lazy-open boundary: opening succeeds while a
				// full integrity check fails. Record it for readability.
				t.Logf("lazy-open-allowed: %q opened successfully but Verify "+
					"rejected the same bytes: %v", tc.name, verifyErr)
			default:
				require.NoError(t, verifyErr,
					"mutation %s: Verify should pass", tc.name)
			}

			runLeafProbe(t, reader, db, probeLeafA, "leafA",
				f.leafPaths["leafA"], tc.leafAErr, tc.skipProbes)
			runLeafProbe(t, reader, db, probeLeafB, "leafB",
				f.leafPaths["leafB"], tc.leafBErr, tc.skipProbes)

			// An address on a normally empty record is either not found or
			// rejected structurally; it must never decode data.
			missing := reader.Lookup(probeMissing)
			if missing.Err() != nil {
				require.True(t, isInvalidDatabase(missing.Err()),
					"missing-IP lookup produced non-structural error: %v",
					missing.Err())
			} else {
				require.False(t, missing.Found())
			}

			runNetworksBounded(t, reader, db)
		})
	}
}

// TestMinimalFixturesAreValid locks the construction baseline: every
// supported record size and both IP versions assemble a database that opens,
// verifies, and satisfies the Decode/DecodePath equivalence property.
func TestMinimalFixturesAreValid(t *testing.T) {
	t.Run("ipv4-record-sizes", func(t *testing.T) {
		for _, recordSize := range []uint{24, 28, 32} {
			t.Run(fmt.Sprintf("rs-%d", recordSize), func(t *testing.T) {
				f := newMinimalFixture(recordSize)
				db := f.assemble(t)
				reader, err := OpenBytes(db.buf)
				require.NoError(t, err)
				t.Cleanup(func() { require.NoError(t, reader.Close()) })
				require.NoError(t, reader.Verify())

				runLeafProbe(t, reader, db, probeLeafA, "leafA",
					f.leafPaths["leafA"], "", false)
				runLeafProbe(t, reader, db, probeLeafB, "leafB",
					f.leafPaths["leafB"], "", false)
				require.False(t, reader.Lookup(probeMissing).Found())

				// Two data-bearing networks, in deterministic DFS order.
				var prefixes []string
				for iter := range reader.Networks() {
					require.NoError(t, iter.Err())
					prefixes = append(prefixes, iter.Prefix().String())
				}
				require.Equal(t, []string{"0.0.0.0/2", "64.0.0.0/2"}, prefixes)
			})
		}
	})

	t.Run("ipv6", func(t *testing.T) {
		f := newIPv6Fixture()
		db := f.assemble(t)
		reader, err := OpenBytes(db.buf)
		require.NoError(t, err)
		t.Cleanup(func() { require.NoError(t, reader.Close()) })
		require.NoError(t, reader.Verify())
		require.Equal(t, uint(96), reader.ipv4Start)

		runLeafProbe(t, reader, db, probeLeafA, "leafA",
			f.leafPaths["leafA"], "", false)
		runLeafProbe(t, reader, db, probeLeafB, "leafB",
			f.leafPaths["leafB"], "", false)

		var prefixes []string
		for iter := range reader.Networks() {
			require.NoError(t, iter.Err())
			prefixes = append(prefixes, iter.Prefix().String())
		}
		require.Equal(t, []string{"0.0.0.0/2", "64.0.0.0/2"}, prefixes)
	})
}

// TestIPv4StartCorruption targets the IPv6-only boundary: Open computes the
// IPv4 subtree start node eagerly but never validates it, while Verify walks
// from the root. A corrupt start node therefore opens lazily but must fail
// every IPv4 access structurally and never return out-of-bounds data.
func TestIPv4StartCorruption(t *testing.T) {
	t.Run("cycle", func(t *testing.T) {
		f := newIPv6Fixture()
		// Point both records of the IPv4 start node back at the node so the
		// IPv4 traversal ends on an internal node regardless of address bits.
		f.nodes[96][0] = nodeRefTo(96)
		f.nodes[96][1] = nodeRefTo(96)
		db := f.assemble(t)

		reader, err := OpenBytes(db.buf)
		require.NoError(t, err) // lazy Open succeeds
		t.Cleanup(func() { require.NoError(t, reader.Close()) })

		verifyErr := reader.Verify()
		require.Error(t, verifyErr)
		require.True(t, isInvalidDatabase(verifyErr))
		t.Logf("lazy-open-allowed: IPv4 start cycle opened but Verify "+
			"rejected: %v", verifyErr)

		result := reader.Lookup(probeLeafA)
		require.Error(t, result.Err())
		require.True(t, isInvalidDatabase(result.Err()))

		// Networks iteration is depth-bounded at 128 bits; it must surface
		// a structural error rather than spin or emit fabricated records.
		runNetworksBounded(t, reader, db)
	})

	t.Run("out-of-bounds-record", func(t *testing.T) {
		f := newIPv6Fixture()
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			// Corrupt only node 97's right record (the 01 route to leafB).
			// The leafA route (00 through node 97's left record) stays intact.
			bad := db.nodeCount + dataSectionSeparatorSize + db.dataLen + 4096
			patchRightRecord(db, 97, bad)
		}
		db := f.assemble(t)

		reader, err := OpenBytes(db.buf)
		require.NoError(t, err)
		t.Cleanup(func() { require.NoError(t, reader.Close()) })
		require.Error(t, reader.Verify())

		result := reader.Lookup(probeLeafB)
		require.ErrorContains(t, result.Err(), "search tree is corrupt")
		require.True(t, isInvalidDatabase(result.Err()))

		// leafA's route stays intact and is still decoded in bounds.
		okResult := reader.Lookup(probeLeafA)
		require.NoError(t, okResult.Err())
		var value any
		require.NoError(t, okResult.Decode(&value))
		assertDecodedWithinBudget(t, db, "leafA", value)

		runNetworksBounded(t, reader, db)
	})
}

// TestPointerFanoutRejected verifies the shared-pointer amplification case:
// the decoder's declared-child reservation rejects the fan-out before it can
// allocate, and Verify reaches the same conclusion. No successful decode may
// materialize the amplified graph.
func TestPointerFanoutRejected(t *testing.T) {
	f := newFanoutFixture()
	db := f.assemble(t)

	reader, err := OpenBytes(db.buf)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, reader.Close()) })

	verifyErr := reader.Verify()
	require.ErrorContains(t, verifyErr, "maximum decoded record size")
	require.True(t, isInvalidDatabase(verifyErr))
	t.Logf("lazy-open-allowed: fanout database opened but Verify rejected: %v",
		verifyErr)

	result := reader.Lookup(probeLeafA)
	require.NoError(t, result.Err()) // tree itself is fine
	var value any
	decodeErr := result.Decode(&value)
	require.ErrorContains(t, decodeErr, "exceeded maximum decoded record size")
	require.True(t, isInvalidDatabase(decodeErr))

	// A bounded traversal observes at most the one network and then the
	// structural decode failure; it never allocates the 2**16 graph.
	runNetworksBounded(t, reader, db)
}

// TestMidIterationError makes the corruption visible only after the iterator
// has already yielded a healthy network, proving the failure is reported
// mid-traversal (not swallowed and not turned into fabricated data).
func TestMidIterationError(t *testing.T) {
	f := newMinimalFixture(24)
	// Corrupt leafB's record (node 1, right) while leaving leafA intact.
	f.rawPatch = func(_ *testing.T, db *assembledDB) {
		bad := db.nodeCount + dataSectionSeparatorSize + db.dataLen + 7
		patchRightRecord(db, 1, bad)
	}
	db := f.assemble(t)

	reader, err := OpenBytes(db.buf)
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, reader.Close()) })
	require.Error(t, reader.Verify())

	sawHealthy := false
	sawError := false
	count := 0
	for iter := range reader.Networks(IncludeNetworksWithoutData()) {
		count++
		require.LessOrEqual(t, count, networksIterationCap)
		if iter.Err() != nil {
			require.True(t, isInvalidDatabase(iter.Err()))
			sawError = true
			break
		}
		if !iter.Found() {
			continue
		}
		var value any
		if err := iter.Decode(&value); err != nil {
			require.True(t, isInvalidDatabase(err))
			sawError = true
			break
		}
		assertDecodedWithinBudget(t, db, iter.Prefix().String(), value)
		sawHealthy = true
	}
	require.True(t, sawHealthy, "iterator must yield the healthy network first")
	require.True(t, sawError, "iterator must then surface the corrupt record")
}

func patchRightRecord(db *assembledDB, node, value uint) {
	mult := db.recordSize / 4
	base := node*mult + mult/2
	switch db.recordSize {
	case 24:
		db.buf[base] = byte(value >> 16)
		db.buf[base+1] = byte(value >> 8)
		db.buf[base+2] = byte(value)
	case 28:
		db.buf[base] = byte(value >> 16)
		db.buf[base+1] = byte(value >> 8)
		db.buf[base+2] = byte(value)
		db.buf[base+3] = (db.buf[base+3] & 0xF0) | byte(value>>24)
	case 32:
		binary.BigEndian.PutUint32(db.buf[base:base+4], uint32(value))
	}
}
