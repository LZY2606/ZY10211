package maxminddb

// Directed fuzzing over structured mutations.
//
// Unlike random byte flips, every fuzz input selects a semantic mutation
// (metadata field, tree record, separator, container length, pointer, or
// payload) and a base fixture. The resulting bytes are driven through the
// same bounded Open/Verify/Lookup/Decode/DecodePath/Networks differential
// harness as TestMutationDifferential. This keeps corpus inputs meaningful
// while letting coverage-guided mutation perturb the mutation parameters.
//
// Complexity: fixtures are constant-size. Each fuzz iteration decodes under
// the library's built-in child-slot/payload budgets and caps Networks at
// fuzzNetworksCap results, so a crafted input cannot turn into a timeout or
// allocation blowup. No network, clock, or filesystem state is used.

import (
	"encoding/binary"
	"net/netip"
	"testing"
)

const fuzzNetworksCap = 256

// fuzzSeeds are directed (mutation-op, parameter) pairs. They cover the
// boundaries the differential tests pin: metadata count mismatch, separator,
// pointer ring, out-of-section pointer, huge container length, and tree
// record corruption.
var fuzzSeeds = [][]byte{
	{0, 0},  // node_count overcount, parameter 0
	{5, 0},  // nonzero separator byte
	{8, 0},  // retarget shared pointer into a ring
	{9, 0},  // shared pointer past the data section
	{10, 0}, // widen the root map length
	{3, 1},  // corrupt a tree record with parameter variant 1
}

// FuzzMutationDifferential applies one structured mutation to a valid
// minimal fixture and asserts the verifier/lazy-decode boundary.
func FuzzMutationDifferential(f *testing.F) {
	for _, seed := range fuzzSeeds {
		f.Add(seed[0], seed[1])
	}
	// A couple of directed raw-byte parameters for the pointer/tree ops.
	f.Add(byte(13), byte(0xFF))
	f.Add(byte(3), byte(7))

	f.Fuzz(func(t *testing.T, op, param byte) {
		fixture := fuzzBuildFixture(op)
		fuzzApplyMutation(t, fixture, op, param)
		db := fixture.assemble(t)

		reader, err := OpenBytes(db.buf)
		if err != nil {
			// Rejected while opening: nothing lazy can disagree with.
			return
		}
		defer func() { _ = reader.Close() }()

		verifyErr := reader.Verify()
		if verifyErr != nil && !isInvalidDatabase(verifyErr) {
			t.Fatalf("Verify returned non-structural error: %v", verifyErr)
		}

		for _, ip := range []netip.Addr{probeLeafA, probeLeafB, probeMissing} {
			result := reader.Lookup(ip)
			if result.Err() != nil {
				if !isInvalidDatabase(result.Err()) {
					t.Fatalf("lookup %s non-structural error: %v", ip, result.Err())
				}
				continue
			}
			if !result.Found() {
				continue
			}

			var value any
			if decErr := result.Decode(&value); decErr != nil {
				if !isInvalidDatabase(decErr) {
					t.Fatalf("decode %s non-structural error: %v", ip, decErr)
				}
			} else {
				assertDecodedWithinBudget(t, db, ip.String(), value)
			}

			// A selected path may fail at a different time than the full
			// decode, but a partial success must remain in bounds and a
			// failure must be structural.
			for _, path := range fixture.leafPaths["leafA"] {
				var byPath any
				if pathErr := result.DecodePath(&byPath, path.elems...); pathErr != nil {
					// Two acceptable outcomes for a corrupted record: a
					// structural InvalidDatabaseError, or a benign path-shape
					// mismatch when the mutation changed a container kind.
					if !isInvalidDatabase(pathErr) &&
						!isPathShapeMismatch(pathErr) {
						t.Fatalf("DecodePath %s unexpected error: %v",
							path.name, pathErr)
					}
					continue
				}
				assertDecodedWithinBudget(t, db, ip.String()+":"+path.name, byPath)
			}
		}

		count := 0
		for iter := range reader.Networks(IncludeNetworksWithoutData()) {
			count++
			if count > fuzzNetworksCap {
				t.Fatalf("Networks exceeded %d results", fuzzNetworksCap)
			}
			if iter.Err() != nil {
				if !isInvalidDatabase(iter.Err()) {
					t.Fatalf("iteration non-structural error: %v", iter.Err())
				}
				continue
			}
			if !iter.Found() {
				continue
			}
			var value any
			if decErr := iter.Decode(&value); decErr != nil {
				if !isInvalidDatabase(decErr) {
					t.Fatalf("iteration decode non-structural error: %v", decErr)
				}
				continue
			}
			assertDecodedWithinBudget(t, db, iter.Prefix().String(), value)
		}
	})
}

// fuzzBuildFixture selects the base fixture by operation family.
func fuzzBuildFixture(op byte) *fixtureSpec {
	switch op {
	case 14, 15: // IPv6 family
		return newIPv6Fixture()
	case 16: // fanout family
		return newFanoutFixture()
	default:
		return newMinimalFixture(24)
	}
}

// fuzzApplyMutation performs exactly one semantic mutation chosen by op.
// Parameter bytes only select among bounded variants; they never control
// allocation sizes directly (lengths are taken from fixed, budget-bounded
// constants), so the fuzzer cannot request an unbounded fixture.
func fuzzApplyMutation(t *testing.T, f *fixtureSpec, op, param byte) {
	t.Helper()
	switch op {
	case 0: // node_count: overcount by 1..8
		count := uint(len(f.nodes))
		f.setMeta("node_count", metaU32(count+uint(param%8)+1))
	case 1: // node_count: undercount, never zero
		count := uint(len(f.nodes))
		if count <= 1 {
			return
		}
		f.setMeta("node_count", metaU32(count-1-uint(param)%(count-1)))
	case 2: // record_size mismatch among 24/28/32
		choices := []uint{24, 28, 32}
		rs := choices[int(param)%len(choices)]
		if rs == f.recordSize {
			rs = 28
		}
		f.setMeta("record_size", metaU16(rs))
	case 3: // corrupt one record in node 1
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			side := param & 1
			value := db.nodeCount + dataSectionSeparatorSize + db.dataLen +
				uint(param%7) + 1
			if side == 0 {
				patchLeftRecord(db, 1, value)
			} else {
				patchRightRecord(db, 1, value)
			}
		}
	case 4: // tree cycle at a low-numbered node
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			node := uint(param % 2)
			patchLeftRecord(db, node, node)
			patchRightRecord(db, node, node)
		}
	case 5: // nonzero separator byte
		pos := int(param) % 16
		f.separator[pos] = 0x01 | (param & 0xFE)
	case 6: // corrupt metadata marker
		f.markerOK = false
	case 7: // truncate the metadata tail by 1..5 bytes
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			cut := int(param%5) + 1
			if cut < len(db.buf) {
				db.buf = db.buf[:len(db.buf)-cut]
			}
		}
	case 8: // retarget the leafA name pointer into a pointer ring
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			at := db.dataStart + db.labelOffset["namePtr:ptr"]
			encodePointer(db.buf, at, db.labelOffset["namePtr:ptr"])
		}
	case 9: // retarget the shared pointer beyond the data section
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			at := db.dataStart + db.labelOffset["namePtr:ptr"]
			db.buf[at] = 0x38
			binary.BigEndian.PutUint32(db.buf[at+1:at+5],
				uint32(db.dataLen+uint(param%32)+1))
		}
	case 10: // widen leafA root map length to a budget-busting constant
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			off := db.dataStart
			v := uint(40_000-285) + uint(param%200)
			db.buf[off] = 7<<5 | 30
			db.buf[off+1] = byte(v >> 8)
			db.buf[off+2] = byte(v)
		}
	case 11: // widen the tags array length via an extended header
		chunk := f.chunks[0]
		wide := uint(40_000-285) + uint(param%200)
		chunk.replaceAtLabel("tagsArr", 1,
			[]byte{0x1E, 0x04, byte(wide >> 8), byte(wide)})
	case 12: // over-long leafB string that runs past its chunk
		if len(f.chunks) >= 2 {
			declared := byte(200 + param%50)
			f.chunks[1].replaceAtLabel("bName", 1,
				[]byte{2<<5 | 29, declared})
		}
	case 13: // corrupt one UTF-8 payload byte
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			targets := []string{"shared", "bName", "cityTokyo"}
			name := targets[int(param)%len(targets)]
			off, ok := db.labelOffset[name]
			if !ok {
				return
			}
			db.buf[db.dataStart+off+1] = 0x80 | (param & 0x3F)
		}
	case 14: // IPv4 start node cycle (IPv6 fixture)
		f.nodes[96][0] = nodeRefTo(96)
		f.nodes[96][1] = nodeRefTo(96)
	case 15: // IPv4 start record out of bounds (IPv6 fixture)
		f.rawPatch = func(_ *testing.T, db *assembledDB) {
			bad := db.nodeCount + dataSectionSeparatorSize + db.dataLen +
				uint(param%64) + 1
			patchRightRecord(db, 97, bad)
		}
	case 16: // fanout fixture needs no extra mutation; it is the case itself
	default: // unknown op: leave the fixture valid, exercising the happy path
	}
}
