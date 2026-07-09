package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.IntFunction;
import java.util.Iterator;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.sandbox.aijoin.AIJoinIndex.PairColumn;
import org.apache.lucene.sandbox.aijoin.AIJoinIndex.SegmentsTuple;
import org.apache.lucene.sandbox.aijoin.AIJoinQuery.JoinSegment;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;


final class AIJoinWeight extends Weight {
  final IndexSearcher maybeStaleJoinSearcher;
  final Map<String, JoinSegment> existingJoinSegments;
  private final IndexReader fromReader;
  private final ScoreMode scoreMode;
  private final float boost;
  private final IndexReader toReader;

  /**
   * [toSegmentOrd][fromSegmentOrd] -> the pair column resolved at construction time; null where
   * the from segment has no cached matches, the pair maps no from-side values, or no cached match
   * falls into the pair's from-doc range. Pair columns record the sidecar segment name, not a
   * leaf context, so they stay valid across join reader refreshes.
   */
  //private final PairColumn[][] pairColumnsByTo;



  AIJoinWeight(AIJoinQuery aiJoinQuery,
    IndexSearcher maybeStaleJoinSearcher,
    Map<String, JoinSegment> existingJoinSegments,
    IndexReader fromReader,
      IndexReader toReader,
      ScoreMode scoreMode,
      float boost)
      throws IOException {
    super(aiJoinQuery);
    //this.aiJoinQuery = aiJoinQuery;
    this.maybeStaleJoinSearcher = maybeStaleJoinSearcher;
    this.existingJoinSegments = existingJoinSegments;
    this.fromReader = fromReader;
    this.scoreMode = scoreMode;
    this.boost = boost;
    this.toReader = toReader;
  }

  /**
   * Sweeps the join index leaves once within a single acquire/release bracket, resolving the
   * requested {@code pairFieldName -> SegmentsTuple} positions into {@link PairColumn}s recorded
   * by sidecar segment name. Returns an array indexed by from-segment ordinal, holding the
   * resolved {@link PairColumn} only for a pair that is both persisted and contributing (its
   * from-doc range holds a cached match); a pair that is unpersisted, or persisted but
   * non-contributing, leaves its slot null. Every pair name found persisted, whether or not it
   * contributes, is added to {@code persistedNames}: the caller must consult that set, not
   * array nullness, to tell an unbuilt pair apart from a persisted-but-non-contributing one, so
   * a persisted-but-non-contributing pair is never rebuilt.
   * @param fromItersBySegOrd from iter pre positioned
   */
  /* private PairColumn[] resolvePairColumns(
      Map<String, SegmentsTuple> pairPositions, Set<String> persistedNames, DocIdSetIterator[] fromItersBySegOrd)
      throws IOException {
    PairColumn[] persisted = new PairColumn[fromReader.leaves().size()];
    IndexSearcher joinSearcher = aiJQuery().joinIndex.acquire();  // it should be propagated from the outer scope
    Set<String> notFound = new HashSet<>(pairPositions.keySet());
    try {
      // the join reader may have been refreshed (e.g. by a concurrent buildPairs) since this
      // weight cached existingJoinSegments against maybeStaleJoinSearcher; a stale map's ords
      // (and, after a drastic change, its segment names) no longer necessarily resolve against
      // the freshly acquired joinSearcher, so it must be recovered first
      Map<String, JoinSegment> currentJoinSegments =
          this.maybeStaleJoinSearcher == joinSearcher
              ? existingJoinSegments
              : ToSegmentJoinContext.recoverExistingJoinSegments(joinSearcher,  existingJoinSegments, pairPositions.keySet());
      for (Iterator<Map.Entry<String, SegmentsTuple>> requestedJoinFieldEntriesIter = pairPositions.entrySet().iterator();
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

          DocIdSetIterator fromSegemtIter = fromItersBySegOrd[position.fromLeafOrd()];
          assert fromSegemtIter != null : "from segment has no cached matches: " + position.fromLeafOrd();
          if (maxFromDoc >=0 && fromSegemtIter.docID() > maxFromDoc) {
            // from iter is already past the last from doc this pair maps, so it cannot contribute
            requestedJoinFieldEntriesIter.remove();
            continue;
          }
          if (minFromDoc >= 0 &&  fromSegemtIter.docID() < minFromDoc) {
            int firstMatch = fromSegemtIter.advance(minFromDoc);
            if (firstMatch == DocIdSetIterator.NO_MORE_DOCS || firstMatch > maxFromDoc) {
              /// wow from iter exhausted, no match in this join segment, so the pair cannot contribute
              // thus we need to return them from request `
              requestedJoinFieldEntriesIter.remove();
              fromItersBySegOrd[position.fromLeafOrd()] = null; // no more matches in this join segment, so the pair cannot contribute
              continue;
            }// else from iter is advanced behind the first from match , good
          }
          persistedNames.add(pairFieldName);// TODO what for?
          persisted[position.fromLeafOrd()] =
              new PairColumn(
                  pairFieldName,
                  joinSegment.joinSegmentName(),
                  maxFromDoc,// i need to score upto
                  ToSegmentJoinContext.loadEdges(joinFeafSeg, AIJoinUtil.TO_EDGES_PREFIX + pairFieldName));
        }
      }

    } finally {
      aiJQuery().joinIndex.release(joinSearcher);
    }
    return persisted;
  } */

  private AIJoinQuery aiJQuery() {
    return (AIJoinQuery) this.parentQuery;
  }

  @Override
  public Explanation explain(LeafReaderContext context, int doc) throws IOException {
    ScorerSupplier supplier = scorerSupplier(context);
    if (supplier != null) {
      Scorer scorer = supplier.get(1);
      if (scorer.iterator().advance(doc) == doc) {
        return Explanation.match(scorer.score(), aiJQuery().toString());
      }
    }
    return Explanation.noMatch(aiJQuery().toString());
  }

  @Override
  public int count(LeafReaderContext context) throws IOException {
    // cost() is a range-size upper bound, not an exact match count, so counting has to
    // fall back to actually driving the two-phase iterator
    return super.count(context);
  }

  @Override
  public ScorerSupplier scorerSupplier(LeafReaderContext tolrc) throws IOException {
    IndexSearcher joinSearcher = aiJQuery().joinIndex.acquire();
    try {
      ToSegmentJoinContext ctx = new ToSegmentJoinContext(tolrc,
       aiJQuery().fromField,
       aiJQuery().fromQuery, aiJQuery().cachedFromSearcher,
       aiJQuery().toField,
       this.toReader,
      this.existingJoinSegments,
      this.maybeStaleJoinSearcher,joinSearcher,
    aiJQuery().joinIndex);
      return ctx.scorerSupplier(scoreMode, boost);
    } finally {
      aiJQuery().joinIndex.release(joinSearcher);
    }






    /* // resolve the pairs already persisted; build the missing ones on demand and resolve them
    // from the refreshed join reader. A pair, once built, always persists at least its edges
    // columns, so missing pairs are built at most once
    Set<String> persistedNames = new HashSet<>();
    PairColumn[] persisted = resolvePairColumns(needFields, persistedNames, fromItersBySegOrd);
    Map<String, SegmentsTuple> missingPairs = new HashMap<>(needFields);
    missingPairs.keySet().removeAll(persistedNames);
    if (!missingPairs.isEmpty()) {
      aiJQuery().joinIndex.writeJoinSegments(missingPairs, fromReader, aiJQuery().fromField, toReader, aiJQuery().toField);
      Set<String> builtNames = new HashSet<>();
      PairColumn[] built = resolvePairColumns(missingPairs, builtNames, fromItersBySegOrd);
      assert builtNames.containsAll(missingPairs.keySet())
          : "pairs still missing after build: " + missingPairs.keySet();
      // put just-built contributing columns into the earlier-seen array; a built pair that
      // still doesn't contribute leaves its slot null, same as an already-persisted one
      for (SegmentsTuple position : missingPairs.values()) {
        if (built[position.fromLeafOrd()] != null) {
          persisted[position.fromLeafOrd()] = built[position.fromLeafOrd()];
        }
      }
    }


    // first pass: union the contributing pairs' to-doc ranges; every possible match in this
    // to segment falls into [minToDoc, maxToDoc]
    int minToDoc = DocIdSetIterator.NO_MORE_DOCS;
    int maxToDoc = -1;
    long matchedFromDocsCount = 0;
    FixedBitSet rangeBits = new FixedBitSet(tolrc.reader().maxDoc() //+ 1//why???
                                          );
    for (int fromOrd = 0; fromOrd < persisted.length; fromOrd++) {
      if (persisted[fromOrd] != null) { // here we can shuffle them, and check whether we have from match on this segement only here
        minToDoc = Math.min(minToDoc, persisted[fromOrd].minToDoc());
        maxToDoc = Math.max(maxToDoc, persisted[fromOrd].maxToDoc());
        matchedFromDocsCount += fromMatches.apply(fromOrd).approximateCardinality();
        rangeBits.set(persisted[fromOrd].minToDoc(), persisted[fromOrd].maxToDoc() + 1);
      }
    }
    if (maxToDoc < 0) {
      // no from segment contributes to this to segment
      return null;
    }
    final int firstToDoc = minToDoc;
    final int lastToDoc = maxToDoc;
    // the first matches() call sweeps every matched from doc to materialize the mapping;
    // later calls are a bitset lookup
    final float matchCost = matchedFromDocsCount; */

  }



  /**
   * Second phase of the join: unions the to-side doc ids mapped from every matched from doc. The
   * returned bitset spans only [minToDoc, maxToDoc], with minToDoc shifted to bit 0; matches
   * mapped below minToDoc are dropped, so the caller must never look them up.
   *
   * <p>All join index reads happen within one acquire/release bracket: the join reader may have
   * been refreshed since weight creation, so each pair column is re-resolved by sidecar segment
   * name in the freshly acquired reader.
   * @param fromIters

   */
  /* private FixedBitSet resolveMatchedToDocs(DocIdSetIterator[] fromIters, PairColumn[] pairColumns, int minToDoc, int maxToDoc)
      throws IOException {
    FixedBitSet matchedToDocs = new FixedBitSet(maxToDoc - minToDoc + 1);
    IndexSearcher joinSearcher = aiJQuery().joinIndex.acquire();
    try {
      Map<String, LeafReaderContext> joinLeavesBySegment = new HashMap<>();
      // we don't need all them only a certain tos
      for (LeafReaderContext joinContext : joinSearcher.getIndexReader().leaves()) {
        joinLeavesBySegment.put(AIJoinUtil.segmentName(joinContext), joinContext);
      }
      for (int fromOrd = 0; fromOrd < pairColumns.length; fromOrd++) {
        PairColumn pairColumn = pairColumns[fromOrd];
        if (pairColumn == null) {
          // this from segment cannot contribute to this to segment
          continue;
        }
        if (fromIters[fromOrd]==null){
          continue; // this from segment has no cached matches, so it cannot contribute to this to segment
        }
        LeafReaderContext joinContext = joinLeavesBySegment.get(pairColumn.joinSegmentName());
        if (joinContext == null) {
          // the join index is append-only: a resolved pair references live side
          // segments, whose sidecar segments reaping must never drop
          throw new IllegalStateException(
              "join index segment ["
                  + pairColumn.joinSegmentName()
                  + "] carrying pair ["
                  + pairColumn.pairFieldName()
                  + "] disappeared");
        }

//        BitSet matches = fromMatches.apply(fromOrd);
        SortedNumericDocValues toDocsByFromDoc =
            joinContext.reader().getSortedNumericDocValues(pairColumn.pairFieldName());

        DocIdSetIterator matchedFromDocs = fromIters[fromOrd];
        for (int fromDoc = matchedFromDocs.docID(); // prepositioned to the first match
            fromDoc != DocIdSetIterator.NO_MORE_DOCS;
            fromDoc = matchedFromDocs.nextDoc()) {

          if (toDocsByFromDoc.advanceExact(fromDoc)) {
            for (int i = 0; i < toDocsByFromDoc.docValueCount(); i++) {
              int toDocMatch = (int) toDocsByFromDoc.nextValue();
              if (toDocMatch >= minToDoc) {
                // matches below minToDoc (including the -1 no-match marker) are
                // unreachable and dropped
                assert toDocMatch <= maxToDoc
                    : "to doc " + toDocMatch + " above edges union max " + maxToDoc;
                matchedToDocs.set(toDocMatch - minToDoc);
              }
            }
          }
        }
      }
    } finally {
      aiJQuery().joinIndex.release(joinSearcher);
    }
    return matchedToDocs;
  } */

  @Override
  public boolean isCacheable(LeafReaderContext lrc) {
    // matches depend on the from-side searcher and the external join index, which the
    // query cache cannot see
    return false;
  }
}
