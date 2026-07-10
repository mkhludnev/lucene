package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

import java.util.concurrent.ThreadLocalRandom;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.sandbox.aijoin.AIJoinIndex.PairColumn;
import org.apache.lucene.sandbox.aijoin.AIJoinIndex.SegmentsTuple;
import org.apache.lucene.sandbox.aijoin.AIJoinQuery.JoinSegmentReference;
import org.apache.lucene.sandbox.aijoin.AIJoinUtil.DocEdges;
import org.apache.lucene.sandbox.aijoin.AIJoinUtil.DocMapping;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;

/**
 * TODO prune by "to" range if it's under slice searching
 * TODO pass raw cacheless searcher
*/
class ToLeafJoinContext {
  final LeafReaderContext toContext;
  final Query fromQuery;
  final IndexSearcher cachedFromSearcher;
  private final IndexSearcher weightAgeJoinSearcher;
  private final IndexSearcher scorerSupplierAgeJoinSearcher;
  final String fromField;
  final String toField;
  final private AIJoinIndex joinIndex;

// TODO all of these might be final since they are set in the constructor
  private int firstToDoc = DocIdSetIterator.NO_MORE_DOCS;
  private int lastToDoc = -1;
  private long matchedToDocsCount=0;
  private FixedBitSet toApproximation=null;
  // consolidated per-pair state; NOT populated yet, wiring the rest of this class onto it is TODO.
  // toCount-descending order is also TODO -- for now cells only ever land at the tail via addJoinCell.
  private final List<JoinCell> joinCells = new ArrayList<>();
  // secondary indices over joinCells, kept in sync by addJoinCell/removeJoinCell: every cell is
  // reachable both by its from-segment ordinal (dense, so a plain array) and by its pair field
  // name (sparse across the full from-segment space, so a map)
  private final JoinCell[] joinCellsByFromSegOrd;
  private final Map<String, JoinCell> joinCellsByPairFieldName = new HashMap<>();

  /** represents a cell in the join matrix bounded to from and to segments */
  class JoinCell{
    final String pairFieldName;
    final DocIdSetIterator fromSegmentDocIdIter;
    final JoinSegmentReference weightAgeJoinSegment;
    final SegmentsTuple segmentsFromTo;
    DocMapping docMapping;
    PairColumn pairColumn;

    JoinCell(
        String pairFieldName,
        SegmentsTuple segmentsFromTo,
        DocIdSetIterator fromSegmentDocIdIter,
        JoinSegmentReference weightAgeJoinSegment) {
      this.pairFieldName = pairFieldName;
      this.segmentsFromTo = segmentsFromTo;
      this.fromSegmentDocIdIter = fromSegmentDocIdIter;
      this.weightAgeJoinSegment = weightAgeJoinSegment;
    }
  }

  /**
   * Appends a new cell for {@code pairFieldName}/{@code segments} to {@link #joinCells} and
   * registers it in both indices.
   */
  private JoinCell addJoinCell(
      String pairFieldName,
      SegmentsTuple segments,
      DocIdSetIterator fromSegmentDocIdIter,
      JoinSegmentReference weightAgeJoinSegment) {
    assert fromSegmentDocIdIter.docID() != DocIdSetIterator.NO_MORE_DOCS;
    assert fromSegmentDocIdIter.docID() >= 0;
    JoinCell cell = new JoinCell(pairFieldName, segments, fromSegmentDocIdIter, weightAgeJoinSegment);
    joinCells.add(cell);
    joinCellsByFromSegOrd[segments.fromLeafOrd()] = cell;
    joinCellsByPairFieldName.put(pairFieldName, cell);
    return cell;
  }

  /**
   * Drops {@code cell} from {@link #joinCells} and both indices, so a pair found to no longer
   * contribute disappears from every view of it at once.
   */
  private void removeJoinCell(JoinCell cell) {
    joinCells.remove(cell);
    joinCellsByFromSegOrd[cell.segmentsFromTo.fromLeafOrd()] = null;
    joinCellsByPairFieldName.remove(cell.pairFieldName);
  }

  /** Looks up a cell by from-segment ordinal, or {@code null} if none is registered there. */
  JoinCell cellByFromSegOrd(int fromSegOrd) {
    return joinCellsByFromSegOrd[fromSegOrd];
  }

  /** Looks up a cell by pair field name, or {@code null} if none is registered under that name. */
  JoinCell cellByPairFieldName(String pairFieldName) {
    return joinCellsByPairFieldName.get(pairFieldName);
  }

  /**
   * A lazy view of every cell's weight-age {@link JoinSegmentReference} as a (pairFieldName, segment)
   * entry, skipping cells without one -- equivalent to iterating a {@code Map<String,
   * JoinSegment>} of just the weight-age segments, but read off {@link JoinCell#weightAgeJoinSegment}
   * instead of keeping a separate map around.
   */
  private Iterable<Map.Entry<String, JoinSegmentReference>> weightAgeJoinSegmentEntries() {
    return () ->
        joinCellsByPairFieldName.entrySet().stream()
            .filter(e -> e.getValue().weightAgeJoinSegment != null)
            .<Map.Entry<String, JoinSegmentReference>>map(
                e -> Map.entry(e.getKey(), e.getValue().weightAgeJoinSegment))
            .iterator();
  }

  /**
   *
   * @param weightAgeJoinSegments the join segments cached at {@link AIJoinQuery#createWeight} time
   * @param weightAgeJoinSearcher the join searcher cached at {@link AIJoinQuery#createWeight} time
   * @param scorerSupplierAgeJoinSearcher the join searcher used at {@link AIJoinWeight#scorerSupplier} time
   */

  ToLeafJoinContext( LeafReaderContext toContext, String fromField,
     Query fromQuery, IndexSearcher cachedFromSearcher, String toField,
      IndexReader toReader, Map<String, JoinSegmentReference> weightAgeJoinSegments,
       IndexSearcher weightAgeJoinSearcher, IndexSearcher scorerSupplierAgeJoinSearcher,
       AIJoinIndex joinIndex) throws IOException {
    this.toContext = toContext;
    this.fromField = fromField;
    this.fromQuery = fromQuery;
    this.cachedFromSearcher = cachedFromSearcher;
    this.toField = toField;
    //this.weightAgeJoinSegments = weightAgeJoinSegments;
    this.weightAgeJoinSearcher = weightAgeJoinSearcher;
    this.scorerSupplierAgeJoinSearcher = scorerSupplierAgeJoinSearcher;
    this.joinIndex = joinIndex;
    this.joinCellsByFromSegOrd = new JoinCell[this.cachedFromSearcher.getLeafContexts().size()];


    stepFromSegIters(weightAgeJoinSegments);

    // it fills each cell's pairColumn
    Map<String, SegmentsTuple> notFound = loadJoinSegments();
    if (!notFound.isEmpty()) { // brace yourself, we need to write'em
        Map<String, DocMapping> written = this.joinIndex.writeJoinSegments(notFound, cachedFromSearcher.getIndexReader()
        , this.fromField, toReader, this.toField);
        assert written.keySet().containsAll(notFound.keySet());
        assert notFound.keySet().containsAll(written.keySet());
        for (Map.Entry<String, DocMapping> entry : written.entrySet()) {
          cellByPairFieldName(entry.getKey()).docMapping = entry.getValue();
        }
        moveFromItersToFirstMatch();
    }
    // now let's read each cell's pairColumn, then docMapping, and build "to" side bitset of approximation

    // first pass: union the contributing pairs' to-doc ranges; every possible match in this
    // to segment falls into [minToDoc, maxToDoc]
    for (JoinCell cell : joinCells) {
      String pairFieldName = cell.pairFieldName;
      {
        PairColumn pairColumn = cell.pairColumn;
        if (pairColumn !=null) {
          // this join segment was not found in the join index, so it was built on demand
          // and is now available via cell.docMapping
          firstToDoc = Math.min(firstToDoc, pairColumn.toDocEdges()[0]);
          lastToDoc = Math.max(lastToDoc, pairColumn.toDocEdges()[1]);
          matchedToDocsCount += pairColumn.toDocEdges()[1]-pairColumn.toDocEdges()[0]+1;
          if (toApproximation == null) {
            toApproximation = new FixedBitSet(toContext.reader().maxDoc());
          }
          toApproximation.set(pairColumn.toDocEdges()[0], pairColumn.toDocEdges()[1] + 1);
          continue;
        } // now fall back to fresh writes
      }
      {
        DocMapping docMapping = cell.docMapping;
        if (docMapping != null) {
          firstToDoc = Math.min(firstToDoc, docMapping.toDocEdges()[0]);
          lastToDoc = Math.max(lastToDoc, docMapping.toDocEdges()[1]);
          matchedToDocsCount += docMapping.toDocEdges()[1]-docMapping.toDocEdges()[0]+1;
          if (toApproximation == null) {
            toApproximation = new FixedBitSet(toContext.reader().maxDoc());
          }
          toApproximation.set(docMapping.toDocEdges()[0], docMapping.toDocEdges()[1] + 1);
          continue;
        }
      }
      throw new IllegalStateException("join segment not found in loaded or new join segments: " + pairFieldName);
    }
  }

  private void moveFromItersToFirstMatch() throws IOException {
    // iterate a snapshot: a dead pair's cell is dropped from joinCells by removeJoinCell below
    for (JoinCell cell : new ArrayList<>(joinCells)) {
      DocMapping docMapping = cell.docMapping;
      if (docMapping == null) {
        continue; // pair was already found on disk, not freshly built this round
      }
      DocIdSetIterator fromSegemtIter = cell.fromSegmentDocIdIter;
      int[] fromDocEdges = docMapping.fromDocEdges();
      int minFromDoc = fromDocEdges[0];
      int maxFromDoc = fromDocEdges[1];
      if (minFromDoc < 0) {
        // {-1, -1} sentinel: this pair maps no from doc to any to doc at all
        removeJoinCell(cell);
        continue;
      }
      if (maxFromDoc >=0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
        && fromSegemtIter.docID() > maxFromDoc) {
        // from iter is already past the last from doc this pair maps, so it cannot contribute
        removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
        continue;
      }
      if (minFromDoc >= 0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
        &&  fromSegemtIter.docID() < minFromDoc) {
        int firstMatch = fromSegemtIter.advance(minFromDoc);
        if (firstMatch == DocIdSetIterator.NO_MORE_DOCS || firstMatch > maxFromDoc) {
          /// wow from iter exhausted, no match in this join segment, so the pair cannot contribute
          // thus we need to return them from request `
          removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
          continue;
        }// else from iter is advanced behind the first from match , good
      }
    }
  }

  /**
   * looks up every {@link JoinCell} in {@link #joinCells} in the join index and loads join segment
   * edges, returning the loaded {@link PairColumn}s
   * SIDE EFFECT: it might drop cells from {@link #joinCells}, if "from" seachers exceeded
   * it returns not found but needed join segs, whihc needs to be written
   * @return the set of required join segments that were not found in the join index
   * @throws IOException
  */
  private Map<String, SegmentsTuple> loadJoinSegments() throws IOException {
    final Map<String, SegmentsTuple> notFound = new HashMap<>(joinCells.size());
    for (JoinCell cell : joinCells) {
      notFound.put(cell.pairFieldName, cell.segmentsFromTo);
    }
    // the join reader may have been refreshed (e.g. by a concurrent buildPairs) since this
    // weight cached existingJoinSegments against maybeStaleJoinSearcher; a stale map's ords
    // (and, after a drastic change, its segment names) no longer necessarily resolve against
    // the freshly acquired joinSearcher, so it must be recovered first
    // recovered lazily below only if the join searcher moved on since weight creation; when it
    // didn't, each pair's weight-age JoinSegment is read straight off its cell instead
    Map<String, JoinSegmentReference> recoveredJoinSegments =
        this.weightAgeJoinSearcher == scorerSupplierAgeJoinSearcher
            ? null
            : ToLeafJoinContext.recoverExistingJoinSegments(scorerSupplierAgeJoinSearcher,  weightAgeJoinSegmentEntries(), joinCellsByPairFieldName.keySet());
    // iterate a snapshot: a dead pair's cell is dropped from joinCells by removeJoinCell below
    for (JoinCell cell : new ArrayList<>(joinCells)) {
      String pairFieldName = cell.pairFieldName;

      JoinSegmentReference joinSegment =
          recoveredJoinSegments != null ? recoveredJoinSegments.get(pairFieldName) : cell.weightAgeJoinSegment;
      if (joinSegment != null) {
        notFound.remove(pairFieldName);
        LeafReaderContext joinFeafSeg = scorerSupplierAgeJoinSearcher.getLeafContexts().get(joinSegment.joinSegmentLeafOrd()); // validate the join segment is still present
        assert AIJoinUtil.segmentName(joinFeafSeg).equals(joinSegment.joinSegmentName());

        int[] fromDocEdges = ToLeafJoinContext.loadEdges(joinFeafSeg, AIJoinUtil.FROM_EDGES_PREFIX + pairFieldName);
        int minFromDoc = fromDocEdges[0];
        int maxFromDoc = fromDocEdges[1];
        // check if from matche hits this join segment

        DocIdSetIterator fromSegemtIter = cell.fromSegmentDocIdIter;
        if (minFromDoc < 0) {
          // {-1, -1} sentinel: this pair maps no from doc to any to doc at all
          removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
          continue;
        }
        if (maxFromDoc >=0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
          && fromSegemtIter.docID() > maxFromDoc) {
          // from iter is already past the last from doc this pair maps, so it cannot contribute
          removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
          continue;
        }
        if (minFromDoc >= 0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
          &&  fromSegemtIter.docID() < minFromDoc) {
          int firstMatch = fromSegemtIter.advance(minFromDoc);
          if (firstMatch == DocIdSetIterator.NO_MORE_DOCS || firstMatch > maxFromDoc) {
            /// wow from iter exhausted, no match in this join segment, so the pair cannot contribute
            // thus we need to return them from request `
            removeJoinCell(cell); // no more matches in this join segment, so the pair cannot contribute
            continue;
          }// else from iter is advanced behind the first from match , good
        }
        cell.pairColumn = // ok. this one is ready for search.
            new PairColumn(
                pairFieldName,
                joinSegment.joinSegmentName(),
                fromDocEdges,
                ToLeafJoinContext.loadEdges(joinFeafSeg, AIJoinUtil.TO_EDGES_PREFIX + pairFieldName),
                // TODO use it for ordefing join segment iteration, desc
                ToLeafJoinContext.loadEdges(joinFeafSeg, AIJoinUtil.TO_COUNT_PREFIX + pairFieldName)[0]
              );
      }
    }
    return notFound;
  }

  /**
   * every to segment call for prepositioned from seg iters
   * @param weightAgeJoinSegments
   * @throws IOException
   * populates a {@link JoinCell} per contributing from segment, its iterator PREPOSITIONED to the
   * first matching doc
   */
  private void stepFromSegIters(Map<String, JoinSegmentReference> weightAgeJoinSegments) throws IOException {
    // TODO peek in underneath searcher cache
    Weight cachedFromWeight = this.cachedFromSearcher.createWeight(this.fromQuery,
        ScoreMode.COMPLETE_NO_SCORES, 1);

    List<LeafReaderContext> leaves = new ArrayList<>(this.cachedFromSearcher.getLeafContexts());
    Collections.shuffle(leaves, ThreadLocalRandom.current());

    for (LeafReaderContext fromContext :
            leaves) {
      //
      ScorerSupplier fromSupplier = cachedFromWeight.scorerSupplier(fromContext);
      if (fromSupplier == null) {
        continue; // no from-side matches in this segment
      }
      Scorer fromScorer = fromSupplier.get(Long.MAX_VALUE);
      DocIdSetIterator matchedFromDocs = fromScorer.iterator();
      Bits liveDocs = fromContext.reader().getLiveDocs();
      if (liveDocs != null) {
        // the cached weight's scorer doesn't filter deletions itself, and a from doc deleted
        // since the pair columns were built (e.g. by an update) must not resolve to a match
        matchedFromDocs =
            new FilteredDocIdSetIterator(matchedFromDocs) {
              @Override
              protected boolean match(int doc) {
                return liveDocs.get(doc);
              }
            };
      }
      if (matchedFromDocs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
        // name every contributing (from, to) pair column; pair field names are unique across pairs
        String pairFieldName =
            AIJoinUtil.pairFieldName(fromContext, this.fromField, toContext, this.toField);
        addJoinCell(
            pairFieldName,
            new SegmentsTuple(fromContext.ord, toContext.ord),
            matchedFromDocs,
            weightAgeJoinSegments.get(pairFieldName));
      }
    }
  }

  /**
   * TODO move to util or index
   * Recovers {@link #weightAgeJoinSegments}, cached against {@link #weightAgeJoinSearcher}, into
   * the freshly acquired {@code joinSearcher}. The join index is append-only under {@link
   * org.apache.lucene.index.NoMergePolicy}, so a seen segment normally still sits at the same
   * leaf ordinal; that's checked first since it's a plain array lookup. A segment reordered since
   * (e.g. by a differently constructed refresh) is found by a name scan instead. If even that
   * fails for any previously seen segment, the join searcher changed too drastically to patch up
   * incrementally, so the whole map is rehashed from scratch via {@link
   * AIJoinQuery#extractExistingJoinColumns}.
   * @param neededKeys required column names on regeneration case
   */
  static Map<String, JoinSegmentReference> recoverExistingJoinSegments(IndexSearcher joinSearcher,
    Iterable<Map.Entry<String, JoinSegmentReference>> existingJoinSegments, Set<String> neededKeys) {
    List<LeafReaderContext> newLeaves = joinSearcher.getLeafContexts();
    Map<String, LeafReaderContext> newLeavesByName = null; // built lazily, only if an ord lookup misses
    Map<String, JoinSegmentReference> recovered = new HashMap<>();
    for (Map.Entry<String, JoinSegmentReference> oldEntry : existingJoinSegments) {
      String pairFieldName = oldEntry.getKey();
      JoinSegmentReference oldSegment = oldEntry.getValue();

      LeafReaderContext byOrd =
          oldSegment.joinSegmentLeafOrd() < newLeaves.size()
              ? newLeaves.get(oldSegment.joinSegmentLeafOrd())
              : null;
      if (byOrd != null && AIJoinUtil.segmentName(byOrd).equals(oldSegment.joinSegmentName())) {
        recovered.put(pairFieldName, new JoinSegmentReference(pairFieldName, oldSegment.joinSegmentName(), byOrd.ord));
        continue;
      }
      // hell, the ord have changed!!, trying to get by name
      if (newLeavesByName == null) {
        newLeavesByName = new HashMap<>(newLeaves.size());
        for (LeafReaderContext leaf : newLeaves) {
          newLeavesByName.put(AIJoinUtil.segmentName(leaf), leaf);
        }
      }
      LeafReaderContext byName = newLeavesByName.get(oldSegment.joinSegmentName());
      if (byName != null) {
        recovered.put(pairFieldName, new JoinSegmentReference(pairFieldName, oldSegment.joinSegmentName(), byName.ord));
        continue;
      }

      // a previously seen join segment vanished outright: the append-only invariant this
      // recovery leans on no longer holds, so patching entry by entry isn't trustworthy anymore
      // -- rehash the whole map from scratch instead of returning a partially-recovered one
      return AIJoinQuery.extractExistingJoinColumns(joinSearcher, neededKeys::contains);
    }
    return recovered;
  }

  /** Reads a pair's persisted values, all stored on doc 0 of the column.
   * TODO move it somewhere
  */
  static int[] loadEdges(LeafReaderContext joinContext, String edgesFieldName)
      throws IOException {
    SortedNumericDocValues edgesDV =
        joinContext.reader().getSortedNumericDocValues(edgesFieldName);
    assert edgesDV != null : "expected edges column to be present: " + edgesFieldName;
    int zeroDoc = edgesDV.nextDoc();
    assert zeroDoc == 0
        : "expected edges column to be fully materialized, but got doc " + zeroDoc;
    int[] values = new int[edgesDV.docValueCount()];
    for (int i = 0; i < values.length; i++) {
      values[i] = (int) edgesDV.nextValue();
    }
    return values;
  }

  public ScorerSupplier scorerSupplier(ScoreMode scoreMode, float boost) {
    if (toApproximation == null || matchedToDocsCount == 0) {
      return null; // no matches in this to segment
    }
    return new ScorerSupplier() {
      @Override
      public Scorer get(long leadCost) throws IOException {
        DocIdSetIterator approximation =
            new BitSetIterator(toApproximation, matchedToDocsCount

            );
        TwoPhaseIterator twoPhase =
            new TwoPhaseIterator(approximation) {
              boolean pruned = false;

              @Override
              public boolean matches() throws IOException {
                if (!pruned) {
                  FixedBitSet matchedToDocs;
                  int shift;
                  // prune the approximation to the resolved matches: drop the
                  // remaining range bits and or the matches back in at their
                  // absolute positions, so the approximation stops visiting
                  // non-matching docs
                  shift = approximation.docID();
                  // TODO the this is: it's enough to just confirm the match the return the control.
                  // TODO there, should be a refined bitset
                  // TODO when we need to refine an approx, we go throug join segments,
                  // TODO and drop them into refined bitset until we confirm the match
                  // TODO the order of iteration is: to side doc freq, which we neeed to persist and read then
                  // TODO once we drop a join segment to refined bitset we exclude it from iterations
                  // TODO also, just boundary check the following matches checks
                  // TODO the following matches() checks, at first confirm with refied bitset,
                  // TODO if it's false procede with to bitset dumping into refined bitset
                  matchedToDocs = refineToMatches( shift);
                  toApproximation.clear(shift, lastToDoc + 1);
                  FixedBitSet.orRange(matchedToDocs, 0, toApproximation, shift, lastToDoc - shift + 1);
                  pruned = true;
                  return toApproximation.get(approximation.docID());
                }
                // the bitset spans [shift, lastToDoc] shifted to zero
                // return matchedToDocs.get(approximation.docID() - shift);
                assert /*return*/ toApproximation.get(approximation.docID()); // always true ??
                return true;
              }

              @Override
              public float matchCost() {
                return matchedToDocsCount;
              }
            };
        return new ConstantScoreScorer(boost, scoreMode, twoPhase);
      }

      @Override
      public long cost() {
        return lastToDoc - firstToDoc + 1;
      }
    };
  }

  /**
  */
  private FixedBitSet refineToMatches(int shift)
      throws IOException {
    FixedBitSet matchedToDocs = new FixedBitSet(lastToDoc - shift + 1);
    IndexSearcher freshSearcher = this.joinIndex.acquire();
    try {

      // recovered lazily below only if the join searcher moved on since scorerSupplier() time;
      // when it didn't, each pair's weight-age JoinSegment is read straight off its cell instead
      Map<String, JoinSegmentReference> recoveredJoinSegments = freshSearcher == this.scorerSupplierAgeJoinSearcher
          ? null
          : ToLeafJoinContext.recoverExistingJoinSegments(freshSearcher, weightAgeJoinSegmentEntries(), joinCellsByPairFieldName.keySet());

      // the recovered map is not scoped to this to-leaf's required fields: it may hold pairs
      // for other to-segments too (superset), and it may still be missing a pair that this
      // context built on demand a moment ago and that only lives in its cell's docMapping so far
      // (subset). Each required field is resolved below by checking both sources in turn, so no
      // set-equality invariant holds here.

      for (JoinCell cell : joinCells) {
        String fieldName = cell.pairFieldName;
        JoinSegmentReference joinSegment = // TODO if ansent check fresh columns from indexer
            recoveredJoinSegments != null ? recoveredJoinSegments.get(fieldName) : cell.weightAgeJoinSegment;
        DocEdges docEdges;
        SortedNumericDocValues toDocsByFromDoc;
        if (joinSegment != null) {
          LeafReaderContext joinContext = freshSearcher.getLeafContexts().get(joinSegment.joinSegmentLeafOrd()); // validate the join segment is still present
          assert fieldName.equals(joinSegment.pairFieldName());
          if (joinContext == null) {
            // the join index is append-only: a resolved pair references live side
            // segments, whose sidecar segments reaping must never drop
            throw new IllegalStateException(
                "join index segment ["
                    + joinSegment.joinSegmentName()
                    + "] carrying pair ["
                    + joinSegment.pairFieldName()
                    + "] disappeared");
          }
          docEdges = cell.pairColumn;
          assert docEdges != null : "join segment edges not loaded for pair: " + fieldName;
          toDocsByFromDoc = joinContext.reader().getSortedNumericDocValues(AIJoinUtil.TO_DOC_VAL_BY_FROM_DOCNUM + joinSegment.pairFieldName());
          // TODO wipe the join cell if its from-iterator is exhausted?
        } else if (cell.docMapping != null) {
          // this join segment was not found in the join index, so it was built on demand and is
          // now available via cell.docMapping
          docEdges = cell.docMapping;
          toDocsByFromDoc = cell.docMapping.toDocByFromDoc();
        } else {
          throw new IllegalStateException("join segment not found in current join segments: " + fieldName);
        }
        dumpToMatches(shift, matchedToDocs, cell, docEdges, toDocsByFromDoc);
      }
    } finally {//TODO release before looping remaining cells separately
      this.joinIndex.release(freshSearcher);
    }
    return matchedToDocs;
  }

  static private void dumpToMatches(int shift, FixedBitSet matchedToDocs, JoinCell cell, DocEdges docEdges, SortedNumericDocValues toDocsByFromDoc) throws IOException {
    DocIdSetIterator matchedFromDocs = cell.fromSegmentDocIdIter;
    for (int fromDoc = matchedFromDocs.docID(); // prepositioned to the first match
        fromDoc != DocIdSetIterator.NO_MORE_DOCS && fromDoc <= docEdges.fromDocEdges()[1]; fromDoc = matchedFromDocs.nextDoc()) {
      if (toDocsByFromDoc.advanceExact(fromDoc)) {
        for (int i = 0; i < toDocsByFromDoc.docValueCount(); i++) {
          int toDocMatch = (int) toDocsByFromDoc.nextValue();
          assert toDocMatch <= docEdges.toDocEdges()[1] && toDocMatch >= docEdges.toDocEdges()[0] : "to doc " + toDocMatch + " above edges union max " + Arrays.toString(docEdges.toDocEdges());
          // shift is wherever the approximation iterator first landed, which need not be
          // the global firstToDoc (e.g. under a boolean conjunction); a match below shift
          // is unreachable -- the iterator only moves forward -- so it's dropped rather
          // than written at a negative offset
          if (toDocMatch >= shift) {
            matchedToDocs.set(toDocMatch - shift);
          }
        }
      }
    }
  }
}
