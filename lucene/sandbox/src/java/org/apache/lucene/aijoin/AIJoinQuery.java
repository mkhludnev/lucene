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
package org.apache.lucene.aijoin;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;

/**
 * Joins the from-side index to the to-side index this query is executed against, resolving
 * from-side docs matching {@code fromQuery} to to-side docs through the auxiliary join index
 * produced by {@link AIJoinUtil#writeAJoinIndex}: there, each (from-segment, to-segment) pair owns
 * a SORTED_NUMERIC column named by both sides' persistent keys, whose doc number is the from-side
 * doc id and whose value is the matching to-side doc id. Matches score a constant.
 */
class AIJoinQuery extends Query {

  /**
   * One from-segment's contribution to the join: the from-side leaf, the join index leaf carrying
   * the pair column, that column's field name, and the cached from-side query matches.
   */
  private record PairColumn(
      LeafReaderContext fromContext,
      LeafReaderContext joinContext,
      String pairFieldName,
      BitSet fromMatches) {}

  private final IndexWriter aJoinWriter;
  //private final IndexReader fromReader;
  private final String fromField;
  private final Query fromQuery;
  private final IndexSearcher fromSearcher;
  private final IndexReader toReader;
  private final String toField;

  AIJoinQuery(
      IndexWriter aJoinWriter,
//      IndexReader fromReader,
      String fromField,
      Query fromQuery,
      IndexSearcher fromSearcher,
      IndexReader toReader,
      String toField) {
    this.aJoinWriter = Objects.requireNonNull(aJoinWriter, "aJoinWriter");
 //   this.fromReader = Objects.requireNonNull(fromReader, "fromReader");
    this.fromField = Objects.requireNonNull(fromField, "fromField");
    this.fromQuery = Objects.requireNonNull(fromQuery, "fromQuery");
    this.fromSearcher = Objects.requireNonNull(fromSearcher, "fromSearcher");
    this.toReader = Objects.requireNonNull(toReader, "toReader");
    this.toField = Objects.requireNonNull(toField, "toField");
  }


  @Override
  public Query rewrite(IndexSearcher indexSearcher) throws IOException {
    // the from-side selection rewrites against the from-side searcher, not against the (to-side)
    // searcher this query is executed with
    Query rewrittenFrom = fromQuery.rewrite(fromSearcher);
    if (rewrittenFrom != fromQuery) {
      return new AIJoinQuery(
          aJoinWriter, /*fromReader,*/ fromField, rewrittenFrom, fromSearcher, toReader, toField);
    }
    return super.rewrite(indexSearcher);
  }

  @Override
  public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
    // run the from-side selection once per weight and cache its matches as one bitset per from
    // segment; from segments without matches keep null
    IndexReader fromReader = fromSearcher.getIndexReader();
    Weight fromWeight =
        fromSearcher.createWeight(
            fromSearcher.rewrite(fromQuery), ScoreMode.COMPLETE_NO_SCORES, 1f);
    BitSet[] fromMatches = new BitSet[fromReader.leaves().size()];
    for (LeafReaderContext fromContext : fromReader.leaves()) {
      ScorerSupplier supplier = fromWeight.scorerSupplier(fromContext);
      if (supplier != null) {
        Scorer scorer = supplier.get(Long.MAX_VALUE);
        fromMatches[fromContext.ord] =
            BitSet.of(scorer.iterator(), fromContext.reader().maxDoc());
      }
    }
    return new Weight(this) {
        @Override
        public Explanation explain(LeafReaderContext lrc, int i) throws IOException {
            throw new UnsupportedOperationException("Not supported yet.");
        }

        @Override
        public ScorerSupplier scorerSupplier(LeafReaderContext tolrc) throws IOException {
            // nocommit: nothing closes this NRT reader yet; lifecycle to be decided
            DirectoryReader joinReader = DirectoryReader.open(aJoinWriter);
            List<PairColumn> pairColumns = new ArrayList<>();
            for (LeafReaderContext fromContext : fromReader.leaves()) {
                // no cached from-side matches means this from segment cannot contribute
                BitSet matches = fromMatches[fromContext.ord];
                if (matches == null) {
                    continue;
                }
                String pairFieldName =
                    AIJoinUtil.pairFieldName(fromContext, fromField, tolrc, toField);
                // find the join index segment carrying this pair's ordinal-map column
                LeafReaderContext joinContext = null;
                for (LeafReaderContext candidate : joinReader.leaves()) {
                    if (candidate.reader().getFieldInfos().fieldInfo(pairFieldName) != null) {
                        joinContext = candidate;
                        break;
                    }
                }
                if (joinContext != null) {
                    pairColumns.add(
                        new PairColumn(fromContext, joinContext, pairFieldName, matches));
                }
                // a missing pair column means the pair was never written (no from-side values);
                // such a from segment cannot contribute matches
            }
            if (pairColumns.isEmpty()) {
                // no from segment contributes matches to this to segment
                joinReader.close();
                return null;
            }
            // union of matched to-side doc ids over all from segments; allocated lazily so null
            // means this to segment got no matches at all
            FixedBitSet matchedToDocs = null;
            for (PairColumn pairColumn : pairColumns) {
                // SortedSetDocValues fromDV =
                //     DocValues.getSortedSet(pairColumn.fromContext().reader(), fromField);
                // FixedBitSet matchedFromOrds = new FixedBitSet((int) fromDV.getValueCount());
                // BitSet fromMatched = pairColumn.fromMatches();
                SortedNumericDocValues toDocsByFromDoc = pairColumn.joinContext().reader().getSortedNumericDocValues(pairColumn.pairFieldName());
                BitSetIterator matchedFromDocs =
                    new BitSetIterator(pairColumn.fromMatches(), pairColumn.fromMatches().approximateCardinality());
                for (int fromDoc = matchedFromDocs.nextDoc();
                    fromDoc != DocIdSetIterator.NO_MORE_DOCS;
                    fromDoc = matchedFromDocs.nextDoc()) {

                    if (toDocsByFromDoc.advanceExact(fromDoc)) {
                        for (int i = 0; i < toDocsByFromDoc.docValueCount(); i++) {
                            int toDocMatch = (int) toDocsByFromDoc.nextValue();
                            if (toDocMatch>=0){
                              if (matchedToDocs == null) {
                                  matchedToDocs = new FixedBitSet(tolrc.reader().maxDoc());
                              }
                              matchedToDocs.set(toDocMatch);
                            }
                        }
                    }
                }
            }
            // the mapping is fully materialized in matchedToDocs; the join index reader is no
            // longer needed by the scorer
            joinReader.close();
            if (matchedToDocs == null) {
                // no from-side match maps into this to segment
                return null;
            }
            final FixedBitSet toDocs = matchedToDocs;
            final long cost = toDocs.cardinality();
            return new ScorerSupplier() {
                @Override
                public Scorer get(long leadCost) throws IOException {
                    return new ConstantScoreScorer(
                        boost, scoreMode, new BitSetIterator(toDocs, cost));
                }

                @Override
                public long cost() {
                    return cost;
                }
            };
        }

        @Override
        public boolean isCacheable(LeafReaderContext lrc) {
            // matches depend on the from-side searcher and the external join index, which the
            // query cache cannot see
            return false;
        }

    };
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
    // readers and the join index writer compare by identity: a reopened reader sees different
    // ordinal spaces, so queries over different reader instances must not be considered equal
    return aJoinWriter == other.aJoinWriter
        && fromSearcher == other.fromSearcher
        && toReader == other.toReader
        && fromField.equals(other.fromField)
        && fromQuery.equals(other.fromQuery)
        && toField.equals(other.toField);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        classHash(),
        System.identityHashCode(aJoinWriter),
        fromField,
        fromQuery,
        System.identityHashCode(fromSearcher),
        System.identityHashCode(toReader),
        toField);
  }
}
