/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TwoPhaseIterator;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/**
 * Joins the from-side index to the to-side index this query is executed against, resolving
 * from-side docs matching {@code fromQuery} to to-side docs through the auxiliary join index
 * managed by {@link AIJoinIndex}: there, each (from-segment, to-segment) pair owns a SORTED_NUMERIC
 * column named by both sides' persistent keys, whose doc number is the from-side doc id and whose
 * value is the matching to-side doc id. Pair columns missing from the join index are built on
 * demand at weight creation, so no explicit build step exists; obtain instances via {@link
 * AIJoinIndex#newJoinQuery}. Matches score a constant.
 */
class AIJoinQuery extends Query {

  private final class AIJoinWeight extends Weight {
    private final BitSet[] fromMatches;
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
    private final PairColumn[][] pairColumnsByTo;

    private AIJoinWeight(
        Query query,
        BitSet[] fromMatches,
        IndexReader fromReader,
        IndexReader toReader,
        ScoreMode scoreMode,
        float boost)
        throws IOException {
      super(query);
      this.fromMatches = fromMatches;
      this.fromReader = fromReader;
      this.scoreMode = scoreMode;
      this.boost = boost;
      this.toReader = toReader;
      this.pairColumnsByTo = new PairColumn[toReader.leaves().size()][fromReader.leaves().size()];
      // name every contributing (from, to) pair column; pair field names are unique across pairs
      Map<String, int[]> pairPositions = new HashMap<>();
      for (LeafReaderContext fromContext : fromReader.leaves()) {
        // no cached from-side matches means this from segment cannot contribute
        if (fromMatches[fromContext.ord] == null) {
          continue;
        }
        for (LeafReaderContext toContext : toReader.leaves()) {
          String pairFieldName =
              AIJoinUtil.pairFieldName(fromContext, fromField, toContext, toField);
          pairPositions.put(pairFieldName, new int[] {toContext.ord, fromContext.ord});
        }
      }
      // resolve the pairs already persisted; build the missing ones on demand and resolve them
      // from the refreshed join reader. A pair, once built, always persists at least its edges
      // columns, so missing pairs are built at most once
      Set<String> persisted = resolvePairColumns(pairPositions);
      Map<String, int[]> missingPairs = new HashMap<>(pairPositions);
      missingPairs.keySet().removeAll(persisted);
      if (!missingPairs.isEmpty()) {
        joinIndex.buildPairs(missingPairs, fromReader, fromField, toReader, toField);
        Set<String> built = resolvePairColumns(missingPairs);
        assert built.containsAll(missingPairs.keySet())
            : "pairs still missing after build: " + missingPairs.keySet();
      }
    }

    /**
     * Sweeps the join index leaves once within a single acquire/release bracket, resolving the
     * requested {@code pairFieldName -> {toSegmentOrd, fromSegmentOrd}} positions into {@link
     * PairColumn}s recorded by sidecar segment name. Returns the pair names found persisted,
     * whether or not they contribute: a persisted pair whose from-doc range holds no cached match
     * keeps a null slot but must not be rebuilt.
     */
    private Set<String> resolvePairColumns(Map<String, int[]> pairPositions) throws IOException {
      Set<String> persisted = new HashSet<>();
      IndexSearcher joinSearcher = joinIndex.acquire();
      try {
        for (LeafReaderContext joinContext : joinSearcher.getIndexReader().leaves()) {
          String segmentName = AIJoinIndex.segmentName(joinContext);
          for (FieldInfo fieldInfo : joinContext.reader().getFieldInfos()) {
            // key off the always-written edges column: the doc-map column itself may be empty
            // when the pair maps no from-side values
            String name = fieldInfo.name;
            if (!name.endsWith(AIJoinUtil.FROM_EDGES_SUFFIX)) {
              continue;
            }
            String pairFieldName =
                name.substring(0, name.length() - AIJoinUtil.FROM_EDGES_SUFFIX.length());
            int[] position = pairPositions.get(pairFieldName);
            if (position == null) {
              continue;
            }
            persisted.add(pairFieldName);
            int fromContextOrd = position[1];
            if (pairColumnsByTo[position[0]][fromContextOrd] != null) {
              continue;
            }
            BitSet fromBits = fromMatches[fromContextOrd];
            int[] fromDocEdges = loadEdges(joinContext, name);
            int minFromDoc = fromDocEdges[0];
            int maxFromDoc = fromDocEdges[1];
            // a pair with no mapped from docs persists sentinel edges that sort to
            // {-1, Integer.MAX_VALUE}; the bounds guard rejects it before nextSetBit
            int firstMatch =
                minFromDoc >= 0 && minFromDoc < fromBits.length()
                    ? fromBits.nextSetBit(minFromDoc)
                    : DocIdSetIterator.NO_MORE_DOCS;
            if (firstMatch != DocIdSetIterator.NO_MORE_DOCS && firstMatch <= maxFromDoc) {
              int[] toDocEdges = loadEdges(joinContext, pairFieldName + AIJoinUtil.TO_EDGES_SUFFIX);
              pairColumnsByTo[position[0]][fromContextOrd] =
                  new PairColumn(pairFieldName, segmentName, toDocEdges[0], toDocEdges[1]);
            }
            // otherwise no cached from-side match falls into the from-doc range this pair
            // maps, so the pair cannot contribute and its slot stays null
          }
        }
      } finally {
        joinIndex.release(joinSearcher);
      }
      return persisted;
    }

    /** Reads a pair's persisted {min, max} doc edges, both stored on doc 0 of the column. */
    private static int[] loadEdges(LeafReaderContext joinContext, String edgesFieldName)
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

    @Override
    public Explanation explain(LeafReaderContext context, int doc) throws IOException {
      ScorerSupplier supplier = scorerSupplier(context);
      if (supplier != null) {
        Scorer scorer = supplier.get(1);
        if (scorer.iterator().advance(doc) == doc) {
          return Explanation.match(scorer.score(), AIJoinQuery.this.toString());
        }
      }
      return Explanation.noMatch(AIJoinQuery.this.toString());
    }

    @Override
    public int count(LeafReaderContext context) throws IOException {
      // cost() is a range-size upper bound, not an exact match count, so counting has to
      // fall back to actually driving the two-phase iterator
      return super.count(context);
    }

    @Override
    public ScorerSupplier scorerSupplier(LeafReaderContext tolrc) throws IOException {
      PairColumn[] pairColumns = pairColumnsByTo[tolrc.ord];
      // first pass: union the contributing pairs' to-doc ranges; every possible match in this
      // to segment falls into [minToDoc, maxToDoc]
      int minToDoc = DocIdSetIterator.NO_MORE_DOCS;
      int maxToDoc = -1;
      long matchedFromDocsCount = 0;
      for (int fromOrd = 0; fromOrd < pairColumns.length; fromOrd++) {
        if (pairColumns[fromOrd] != null) { // here we can shuffle them, and check whether we have from match on this segement only here
          minToDoc = Math.min(minToDoc, pairColumns[fromOrd].minToDoc());
          maxToDoc = Math.max(maxToDoc, pairColumns[fromOrd].maxToDoc());
          matchedFromDocsCount += fromMatches[fromOrd].approximateCardinality();
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
      final float matchCost = matchedFromDocsCount;
      return new ScorerSupplier() {
        @Override
        public Scorer get(long leadCost) throws IOException {
          FixedBitSet rangeBits = new FixedBitSet(lastToDoc + 1);
          rangeBits.set(firstToDoc, lastToDoc + 1); // drop seperate [minTo maxTo] ranges
          DocIdSetIterator approximation =
              new BitSetIterator(rangeBits, lastToDoc - firstToDoc + 1);
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
                    matchedToDocs = resolveMatchedToDocs(pairColumns, shift, lastToDoc);
                    rangeBits.clear(shift, lastToDoc + 1);
                    FixedBitSet.orRange(matchedToDocs, 0, rangeBits, shift, lastToDoc - shift + 1);
                    pruned = true;
                    return rangeBits.get(approximation.docID());
                  }
                  // the bitset spans [shift, lastToDoc] shifted to zero
                  // return matchedToDocs.get(approximation.docID() - shift);
                  assert /*return*/ rangeBits.get(approximation.docID()); // always true ??
                  return true;
                }

                @Override
                public float matchCost() {
                  return matchCost;
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
     * Second phase of the join: unions the to-side doc ids mapped from every matched from doc. The
     * returned bitset spans only [minToDoc, maxToDoc], with minToDoc shifted to bit 0; matches
     * mapped below minToDoc are dropped, so the caller must never look them up.
     *
     * <p>All join index reads happen within one acquire/release bracket: the join reader may have
     * been refreshed since weight creation, so each pair column is re-resolved by sidecar segment
     * name in the freshly acquired reader.
     */
    private FixedBitSet resolveMatchedToDocs(PairColumn[] pairColumns, int minToDoc, int maxToDoc)
        throws IOException {
      FixedBitSet matchedToDocs = new FixedBitSet(maxToDoc - minToDoc + 1);
      IndexSearcher joinSearcher = joinIndex.acquire();
      try {
        Map<String, LeafReaderContext> joinLeavesBySegment = new HashMap<>();
        for (LeafReaderContext joinContext : joinSearcher.getIndexReader().leaves()) {
          joinLeavesBySegment.put(AIJoinIndex.segmentName(joinContext), joinContext);
        }
        for (int fromOrd = 0; fromOrd < pairColumns.length; fromOrd++) {
          PairColumn pairColumn = pairColumns[fromOrd];
          if (pairColumn == null) {
            // this from segment cannot contribute to this to segment
            continue;
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
          BitSet matches = fromMatches[fromOrd];
          SortedNumericDocValues toDocsByFromDoc =
              joinContext.reader().getSortedNumericDocValues(pairColumn.pairFieldName());
          BitSetIterator matchedFromDocs =
              new BitSetIterator(matches, matches.approximateCardinality());
          for (int fromDoc = matchedFromDocs.nextDoc();
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
        joinIndex.release(joinSearcher);
      }
      return matchedToDocs;
    }

    @Override
    public boolean isCacheable(LeafReaderContext lrc) {
      // matches depend on the from-side searcher and the external join index, which the
      // query cache cannot see
      return false;
    }
  }

  /**
   * One (from-segment, to-segment) pair's ordinal-map column: its field name and the name of the
   * sidecar segment carrying it, so the column survives join reader refreshes.
   */
  private record PairColumn(
      String pairFieldName, String joinSegmentName, int minToDoc, int maxToDoc) {}

  private final AIJoinIndex joinIndex;
  private final String fromField;
  private final Query fromQuery;
  private final IndexSearcher fromSearcher;
  private final String toField;

  AIJoinQuery(
      AIJoinIndex joinIndex,
      String fromField,
      Query fromQuery,
      IndexSearcher fromSearcher,
      String toField) {
    this.joinIndex = Objects.requireNonNull(joinIndex, "joinIndex");
    this.fromField = Objects.requireNonNull(fromField, "fromField");
    this.fromQuery = Objects.requireNonNull(fromQuery, "fromQuery");
    this.fromSearcher = Objects.requireNonNull(fromSearcher, "fromSearcher");
    this.toField = Objects.requireNonNull(toField, "toField");
  }

  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    // the from-side selection rewrites against the from-side searcher, not against the (to-side)
    // searcher this query is executed with
    Query rewrittenFrom = fromQuery.rewrite(fromSearcher);
    if (rewrittenFrom != fromQuery) {
      return new AIJoinQuery(joinIndex, fromField, rewrittenFrom, fromSearcher, toField);
    }
    return super.rewrite(indexSearcher);
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost)
      throws IOException {
    // run the from-side selection once per weight and cache its matches as one bitset per from
    // segment; from segments without matches keep null
    IndexReader fromReader = fromSearcher.getIndexReader();
    Weight fromWeight = fromSearcher.createWeight(fromQuery, ScoreMode.COMPLETE_NO_SCORES, 1f);
    BitSet[] fromMatches = new BitSet[fromReader.leaves().size()];
    for (LeafReaderContext fromContext : fromReader.leaves()) {
      ScorerSupplier supplier = fromWeight.scorerSupplier(fromContext);
      if (supplier != null) {
        Scorer scorer = supplier.get(Long.MAX_VALUE);
        DocIdSetIterator iterator = scorer.iterator();
        Bits liveDocs = fromContext.reader().getLiveDocs();
        if (liveDocs != null) {
          // scorers don't filter deletions themselves: that's collection-time acceptDocs work,
          // which this cached-bitset path bypasses
          iterator =
              new FilteredDocIdSetIterator(iterator) {
                @Override
                protected boolean match(int doc) {
                  return liveDocs.get(doc);
                }
              };
        }
        fromMatches[fromContext.ord] = BitSet.of(iterator, fromContext.reader().maxDoc());
      }
    }
    return new AIJoinWeight(
        this, fromMatches, fromReader, searcher.getIndexReader(), scoreMode, boost);
  }

  @Override
  public String toString(String field) {
    return "AIJoinQuery(" + fromField + " -> " + toField + ", from: " + fromQuery + ")";
  }

  @Override
  public void visit(QueryVisitor visitor) {
    visitor.visitLeaf(this);
  }

  @Override
  public boolean equals(Object other) {
    return sameClassAs(other) && equalsTo((AIJoinQuery) other);
  }

  private boolean equalsTo(AIJoinQuery other) {
    // the join index and the from searcher compare by identity: a reopened from reader sees
    // different ordinal spaces, so queries over different searcher instances must not be
    // considered equal
    return joinIndex == other.joinIndex
        && fromSearcher == other.fromSearcher
        && fromField.equals(other.fromField)
        && fromQuery.equals(other.fromQuery)
        && toField.equals(other.toField);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        classHash(),
        System.identityHashCode(joinIndex),
        fromField,
        fromQuery,
        System.identityHashCode(fromSearcher),
        toField);
  }
}
