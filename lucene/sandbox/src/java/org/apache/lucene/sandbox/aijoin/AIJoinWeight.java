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
  }


  @Override
  public boolean isCacheable(LeafReaderContext lrc) {
    // matches depend on the from-side searcher and the external join index, which the
    // query cache cannot see
    return false;
  }
}
