package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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
import org.apache.lucene.sandbox.aijoin.AIJoinQuery.JoinSegment;
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
class ToSegmentJoinContext {
  final LeafReaderContext toContext;
  final Query fromQuery;
  final IndexSearcher cachedFromSearcher;
  private final DocIdSetIterator[] fromScorersBySegOrd;
  private final Map<String, JoinSegment> existingJoinSegments;
  private final IndexSearcher maybeStaleJoinSearcher;
  private final IndexSearcher joinSearcher;
  final String fromField;
  final String toField;
  final Map<String, SegmentsTuple> requiredJoinFields;
  final private AIJoinIndex joinIndex;

// TODO all of these might be final since they are set in the constructor
  private int firstToDoc = DocIdSetIterator.NO_MORE_DOCS;
  private int lastToDoc = -1;
  private long matchedToDocsCount=0;
  private FixedBitSet toApproximation=null;
  private Map<String, DocMapping> newJoinSegments = null;
  private final PairColumn[] loadedJoinSegments;// = new PairColumn[this.cachedFromSearcher.getLeafContexts().size()];

  ToSegmentJoinContext( LeafReaderContext toContext, String fromField,
     Query fromQuery, IndexSearcher cachedFromSearcher, String toField,
      IndexReader toReader, Map<String, JoinSegment> existingJoinSegments,
       IndexSearcher maybeStaleJoinSearcher, IndexSearcher joinSearcher,
       AIJoinIndex joinIndex) throws IOException {
    this.toContext = toContext;
    this.fromField = fromField;
    this.fromQuery = fromQuery;
    this.cachedFromSearcher = cachedFromSearcher;
    this.toField = toField;
    this.existingJoinSegments = existingJoinSegments;
    this.maybeStaleJoinSearcher = maybeStaleJoinSearcher;
    this.joinSearcher = joinSearcher;
    this.joinIndex = joinIndex;
    this.loadedJoinSegments = new PairColumn[this.cachedFromSearcher.getLeafContexts().size()];


    this.fromScorersBySegOrd = stepFromSegIters();
    // it would be nice to trigger the join segment resolving right after pre positionoing from searcher segment
    this.requiredJoinFields = requiredJoinFields();

    newJoinSegments = null;
    // it fills this.loadedJoinSegments
    Map<String, SegmentsTuple> notFound = loadJoinSegments(loadedJoinSegments);
    if (!notFound.isEmpty()) { // brace yourself, we need to write'em
        newJoinSegments = this.joinIndex.writeJoinSegments(notFound, cachedFromSearcher.getIndexReader()
        , this.fromField, toReader, this.toField);
        assert newJoinSegments.keySet().containsAll(notFound.keySet());
        assert notFound.keySet().containsAll(newJoinSegments.keySet());
        moveFromItersToFirstMatch(newJoinSegments);
    }
    // now let's read loadedJoinSegments, then newJoinSegments and buils "to" side bitset of approximation

    // first pass: union the contributing pairs' to-doc ranges; every possible match in this
    // to segment falls into [minToDoc, maxToDoc]
    for (Map.Entry<String, SegmentsTuple> entry : requiredJoinFields.entrySet()) {
      String pairFieldName = entry.getKey();
      SegmentsTuple position = entry.getValue();
      {
        PairColumn pairColumn = loadedJoinSegments[position.fromLeafOrd()];
        if (pairColumn !=null) {
          // this join segment was not found in the join index, so it was built on demand
          // and is now available in newJoinSegments
          ///pairColumn = newJoinSegments.get(pairFieldName);
          firstToDoc = Math.min(firstToDoc, pairColumn.toDocEdges()[0]);
          lastToDoc = Math.max(lastToDoc, pairColumn.toDocEdges()[1]);
          matchedToDocsCount += pairColumn.toDocEdges()[1]-pairColumn.toDocEdges()[0];
          if (toApproximation == null) {
            toApproximation = new FixedBitSet(toContext.reader().maxDoc());
          }
          toApproximation.set(pairColumn.toDocEdges()[0], pairColumn.toDocEdges()[1] + 1);
          continue;
        } // now fall back to fresh writes
      }
      {
        if (newJoinSegments!=null){
          DocMapping found = newJoinSegments.computeIfPresent(pairFieldName, (k, pairColumn) -> {
                //TODO extract/abstract
                firstToDoc = Math.min(firstToDoc, pairColumn.toDocEdges()[0]);
                lastToDoc = Math.max(lastToDoc, pairColumn.toDocEdges()[1]);
                matchedToDocsCount += pairColumn.toDocEdges()[1]-pairColumn.toDocEdges()[0];
                if (toApproximation == null) {
                  toApproximation = new FixedBitSet(toContext.reader().maxDoc());
                }
                toApproximation.set(pairColumn.toDocEdges()[0], pairColumn.toDocEdges()[1] + 1);
                return pairColumn;
              });
          assert found != null : "join segment not found in newJoinSegments: " + pairFieldName;
          continue;
        }
      }
      throw new IllegalStateException("join segment not found in loaded or new join segments: " + pairFieldName);
    }
  }

  private void moveFromItersToFirstMatch(Map<String, DocMapping> newJoinSegments) throws IOException {
    for (Iterator<Map.Entry<String, DocMapping>> iter = newJoinSegments.entrySet().iterator();
        iter.hasNext(); ) {
      Map.Entry<String, DocMapping> entry = iter.next();
      String pairFieldName = entry.getKey();
      SegmentsTuple position = this.requiredJoinFields.get(pairFieldName);
      assert position != null;
      DocIdSetIterator fromSegemtIter = this.fromScorersBySegOrd[position.fromLeafOrd()];
      assert fromSegemtIter != null : "from segment has no cached matches: " + position.fromLeafOrd();
      int[] fromDocEdges = entry.getValue().fromDocEdges();
      int minFromDoc = fromDocEdges[0];
      int maxFromDoc = fromDocEdges[1];
      if (maxFromDoc >=0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
        && fromSegemtIter.docID() > maxFromDoc) {
        // from iter is already past the last from doc this pair maps, so it cannot contribute
        this.fromScorersBySegOrd[position.fromLeafOrd()] = null; // no more matches in this join segment, so the pair cannot contribute
        this.requiredJoinFields.remove(pairFieldName);
        iter.remove();
        continue;
      }
      if (minFromDoc >= 0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
        &&  fromSegemtIter.docID() < minFromDoc) {
        int firstMatch = fromSegemtIter.advance(minFromDoc);
        if (firstMatch == DocIdSetIterator.NO_MORE_DOCS || firstMatch > maxFromDoc) {
          /// wow from iter exhausted, no match in this join segment, so the pair cannot contribute
          // thus we need to return them from request `
          this.fromScorersBySegOrd[position.fromLeafOrd()] = null; // no more matches in this join segment, so the pair cannot contribute
          this.requiredJoinFields.remove(pairFieldName);
          iter.remove();
          continue;
        }// else from iter is advanced behind the first from match , good
      }
    }
  }

  /**
   * looks up requiredJoinFields in existingJoinSegments and loads join segemnt edges returns loaded PairColumn
   * SIDE EFFECT: it might drop fileds from {@code requiredJoinFields}, if "from" seachers exceeded
   * it returns not found but needed join segs, whihc needs to be written
   * @param loadedJoinSegments preallocated array of size fromSearcher.leaves().size() to be filled with loaded pair columns
   * @return the set of required join segments that were not found in the join index
   * @throws IOException
  */
  private Map<String, SegmentsTuple> loadJoinSegments(PairColumn[] loadedJoinSegments) throws IOException {
    final Map<String, SegmentsTuple> notFound = new HashMap<>(this.requiredJoinFields);
    // the join reader may have been refreshed (e.g. by a concurrent buildPairs) since this
    // weight cached existingJoinSegments against maybeStaleJoinSearcher; a stale map's ords
    // (and, after a drastic change, its segment names) no longer necessarily resolve against
    // the freshly acquired joinSearcher, so it must be recovered first
    Map<String, JoinSegment> currentJoinSegments =
        this.maybeStaleJoinSearcher == joinSearcher
            ? existingJoinSegments
            : ToSegmentJoinContext.recoverExistingJoinSegments(joinSearcher,  existingJoinSegments, this.requiredJoinFields.keySet());
    for (Iterator<Map.Entry<String, SegmentsTuple>> requestedJoinFieldEntriesIter = this.requiredJoinFields.entrySet().iterator();
        requestedJoinFieldEntriesIter.hasNext(); ) {
      Map.Entry<String, SegmentsTuple> entry = requestedJoinFieldEntriesIter.next();
      String pairFieldName = entry.getKey();
      SegmentsTuple position = entry.getValue();

      JoinSegment joinSegment = currentJoinSegments.get(pairFieldName);
      if (joinSegment != null) {
        notFound.remove(pairFieldName);
        LeafReaderContext joinFeafSeg = joinSearcher.getLeafContexts().get(joinSegment.joinSegmentLeafOrd()); // validate the join segment is still present
        assert AIJoinUtil.segmentName(joinFeafSeg).equals(joinSegment.joinSegmentName());

        int[] fromDocEdges = ToSegmentJoinContext.loadEdges(joinFeafSeg, AIJoinUtil.FROM_EDGES_PREFIX + pairFieldName);
        int minFromDoc = fromDocEdges[0];
        int maxFromDoc = fromDocEdges[1];
        // check if from matche hits this join segment

        DocIdSetIterator fromSegemtIter = this.fromScorersBySegOrd[position.fromLeafOrd()];
        assert fromSegemtIter != null : "from segment has no cached matches: " + position.fromLeafOrd();
        if (maxFromDoc >=0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
          && fromSegemtIter.docID() > maxFromDoc) {
          // from iter is already past the last from doc this pair maps, so it cannot contribute
          requestedJoinFieldEntriesIter.remove();
          this.fromScorersBySegOrd[position.fromLeafOrd()] = null; // no more matches in this join segment, so the pair cannot contribute
          continue;
        }
        if (minFromDoc >= 0 && maxFromDoc!=DocIdSetIterator.NO_MORE_DOCS
          &&  fromSegemtIter.docID() < minFromDoc) {
          int firstMatch = fromSegemtIter.advance(minFromDoc);
          if (firstMatch == DocIdSetIterator.NO_MORE_DOCS || firstMatch > maxFromDoc) {
            /// wow from iter exhausted, no match in this join segment, so the pair cannot contribute
            // thus we need to return them from request `
            requestedJoinFieldEntriesIter.remove();
            this.fromScorersBySegOrd[position.fromLeafOrd()] = null; // no more matches in this join segment, so the pair cannot contribute
            continue;
          }// else from iter is advanced behind the first from match , good
        }
        loadedJoinSegments[position.fromLeafOrd()] = // ok. this one is ready for search.
            new PairColumn(
                pairFieldName,
                joinSegment.joinSegmentName(),
                maxFromDoc,// i need to score upto
                ToSegmentJoinContext.loadEdges(joinFeafSeg, AIJoinUtil.TO_EDGES_PREFIX + pairFieldName));
      }
    }
    return notFound;
  }

  /**
   * every to segment call for prepositioned from seg iters
   * @throws IOException
   * @returns null or from seg iter PREPOSITIONED to the first matching doc! size it from reader.leaves().size()
   */
  private DocIdSetIterator[] stepFromSegIters() throws IOException {
    DocIdSetIterator[] fromIters = new DocIdSetIterator[this.cachedFromSearcher.getLeafContexts().size()];
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
      if (matchedFromDocs.nextDoc() != DocIdSetIterator.NO_MORE_DOCS)
        fromIters[fromContext.ord] = matchedFromDocs;
        // it's time to touch join segemnt 1
    }
    return fromIters; // TODO
  }

  Map<String, SegmentsTuple> requiredJoinFields() {
        // name every contributing (from, to) pair column; pair field names are unique across pairs
    Map<String, SegmentsTuple> needFields = new HashMap<>(); // to is the same always
    for (LeafReaderContext fromContext : this.cachedFromSearcher.getLeafContexts()) {
      // no cached from-side matches means this from segment cannot contribute
      if (this.fromScorersBySegOrd[fromContext.ord] == null) {
        continue;
      }
      assert this.fromScorersBySegOrd[fromContext.ord].docID() != DocIdSetIterator.NO_MORE_DOCS;
      assert this.fromScorersBySegOrd[fromContext.ord].docID() >= 0;
      String pairFieldName =
          AIJoinUtil.pairFieldName(fromContext, this.fromField, toContext, this.toField);
      needFields.put(pairFieldName, new SegmentsTuple(fromContext.ord, toContext.ord));
    }
    return needFields;
  }

  /**
   * TODO move to util or index
   * Recovers {@link #existingJoinSegments}, cached against {@link #maybeStaleJoinSearcher}, into
   * the freshly acquired {@code joinSearcher}. The join index is append-only under {@link
   * org.apache.lucene.index.NoMergePolicy}, so a seen segment normally still sits at the same
   * leaf ordinal; that's checked first since it's a plain array lookup. A segment reordered since
   * (e.g. by a differently constructed refresh) is found by a name scan instead. If even that
   * fails for any previously seen segment, the join searcher changed too drastically to patch up
   * incrementally, so the whole map is rehashed from scratch via {@link
   * AIJoinQuery#extractExistingJoinColumns}.
   * @param neededKeys required column names on regeneration case
   */
  static Map<String, JoinSegment> recoverExistingJoinSegments(IndexSearcher joinSearcher, Map<String, JoinSegment> existingJoinSegments, Set<String> neededKeys) {
    List<LeafReaderContext> newLeaves = joinSearcher.getLeafContexts();
    Map<String, LeafReaderContext> newLeavesByName = null; // built lazily, only if an ord lookup misses
    Map<String, JoinSegment> recovered = new HashMap<>(existingJoinSegments.size());
    for (Map.Entry<String, JoinSegment> oldEntry : existingJoinSegments.entrySet()) {
      String pairFieldName = oldEntry.getKey();
      JoinSegment oldSegment = oldEntry.getValue();

      LeafReaderContext byOrd =
          oldSegment.joinSegmentLeafOrd() < newLeaves.size()
              ? newLeaves.get(oldSegment.joinSegmentLeafOrd())
              : null;
      if (byOrd != null && AIJoinUtil.segmentName(byOrd).equals(oldSegment.joinSegmentName())) {
        recovered.put(pairFieldName, new JoinSegment(pairFieldName, oldSegment.joinSegmentName(), byOrd.ord));
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
        recovered.put(pairFieldName, new JoinSegment(pairFieldName, oldSegment.joinSegmentName(), byName.ord));
        continue;
      }

      // a previously seen join segment vanished outright: the append-only invariant this
      // recovery leans on no longer holds, so patching entry by entry isn't trustworthy anymore
      // -- rehash the whole map from scratch instead of returning a partially-recovered one
      return AIJoinQuery.extractExistingJoinColumns(joinSearcher, neededKeys::contains);
    }
    return recovered;
  }

  /** Reads a pair's persisted {min, max} doc edges, both stored on doc 0 of the column.
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
    assert edgesDV.docValueCount() == 2
        : "expected edges column to be fully materialized, but got "
            + edgesDV.docValueCount()
            + " values";
    return new int[] {(int) edgesDV.nextValue(), (int) edgesDV.nextValue()};
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

  private FixedBitSet refineToMatches(int shift)
      throws IOException {
    FixedBitSet matchedToDocs = new FixedBitSet(lastToDoc - shift + 1);
    IndexSearcher freshSearcher = this.joinIndex.acquire();
    try {

      Map<String, JoinSegment> currentJoinSegments = freshSearcher == this.joinSearcher
          ? existingJoinSegments
          : ToSegmentJoinContext.recoverExistingJoinSegments(freshSearcher, existingJoinSegments, this.requiredJoinFields.keySet());

      assert currentJoinSegments.keySet().containsAll(this.requiredJoinFields.keySet());
      assert this.requiredJoinFields.keySet().containsAll(currentJoinSegments.keySet());

      for (Map.Entry<String, SegmentsTuple> entry : requiredJoinFields.entrySet()) {
        String fieldName = entry.getKey();
        SegmentsTuple fromToLeaves = entry.getValue();
        JoinSegment joinSegment = currentJoinSegments.get(fieldName); // TODO if ansent check fresh columns from indexer
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
          //loadedJoinSegments[]
          PairColumn joinSegmentEdges = this.loadedJoinSegments[fromToLeaves.fromLeafOrd()];
          assert joinSegmentEdges != null : "join segment edges not loaded for pair: " + fieldName;

          SortedNumericDocValues toDocsByFromDoc = joinContext.reader().getSortedNumericDocValues(AIJoinUtil.TO_DOC_VAL_BY_FROM_DOCNUM + joinSegment.pairFieldName());
          DocIdSetIterator matchedFromDocs = this.fromScorersBySegOrd[this.requiredJoinFields.get(fieldName).fromLeafOrd()];
          assert matchedFromDocs != null : "from segment has no cached matches: " + this.requiredJoinFields.get(fieldName).fromLeafOrd();
          for (int fromDoc = matchedFromDocs.docID(); // prepositioned to the first match
              fromDoc != DocIdSetIterator.NO_MORE_DOCS && fromDoc <= joinSegmentEdges.maxFromDoc(); fromDoc = matchedFromDocs.nextDoc()) {

            if (toDocsByFromDoc.advanceExact(fromDoc)) {
              for (int i = 0; i < toDocsByFromDoc.docValueCount(); i++) {
                int toDocMatch = (int) toDocsByFromDoc.nextValue();
                // matches below minToDoc (including the -1 no-match marker) are
                // unreachable and dropped
                assert toDocMatch <= joinSegmentEdges.toDocEdges()[1] : "to doc " + toDocMatch + " above edges union max " + joinSegmentEdges.toDocEdges()[1];
                matchedToDocs.set(toDocMatch - shift);
              }
            }
          }
          // TODO wipe fromScorersBySegOrd[] is it's exhausted ?
          continue;
        } else if (newJoinSegments != null && newJoinSegments.containsKey(fieldName)) {
          DocMapping docMapping = newJoinSegments.get(fieldName); // TODO extract/abstract

          DocIdSetIterator matchedFromDocs = this.fromScorersBySegOrd[this.requiredJoinFields.get(fieldName).fromLeafOrd()];
          assert matchedFromDocs != null : "from segment has no cached matches: " + this.requiredJoinFields.get(fieldName).fromLeafOrd();
          for (int fromDoc = matchedFromDocs.docID(); // prepositioned to the first match
              fromDoc != DocIdSetIterator.NO_MORE_DOCS && fromDoc <= docMapping.fromDocEdges()[1]; fromDoc = matchedFromDocs.nextDoc()) {
            int toDocMatch;
            if ((toDocMatch = docMapping.toDocByFromDoc()[fromDoc]) >= 0) {

              // matches below minToDoc (including the -1 no-match marker) are
              // unreachable and dropped
              assert toDocMatch <= docMapping.toDocEdges()[1] && toDocMatch >= docMapping.toDocEdges()[0] : "to doc " + toDocMatch + " above edges union max " + Arrays.toString(docMapping.toDocEdges());
              matchedToDocs.set(toDocMatch - shift);
            }
          }
          continue; // this join segment was not found in the join index, so it was built on demand and is now available in newJoinSegments
        }
        throw new IllegalStateException("join segment not found in current join segments: " + fieldName);
      }
    } finally {//TODO release before looping newJoinSegments separately
      this.joinIndex.release(freshSearcher);
    }
    return matchedToDocs;
  }
}
