package org.apache.lucene.sandbox.aijoin;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.column.Column;
import org.apache.lucene.document.column.ColumnBatch;
import org.apache.lucene.document.column.LongColumn;
import org.apache.lucene.document.column.LongTupleCursor;
import org.apache.lucene.document.column.LongColumn.NumericKind;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.SortedNumericDocValues;
import org.apache.lucene.sandbox.aijoin.AIJoinUtil.JoinColumnModel;
import org.apache.lucene.search.DocIdSetIterator;

final class AIJoinColumnWriter extends AIJoinWriter {

  AIJoinColumnWriter() {}

  private static final FieldType TO_DOCS_FIELD_TYPE = new FieldType();

  static {
    TO_DOCS_FIELD_TYPE.setDocValuesType(DocValuesType.SORTED_NUMERIC);
    TO_DOCS_FIELD_TYPE.freeze();
  }

  @Override
  void writeJoinColumns(
      IndexWriter writer, int batchNumDocs, Map<String, JoinColumnModel> mappings) throws IOException {
    List<Column> columns = new ArrayList<>();
    for (Map.Entry<String, JoinColumnModel> entry : mappings.entrySet()) {
      columns.addAll(createJoinColumns(entry.getValue(), entry.getKey()));
    }
    writer.addBatch(
        new ColumnBatch() {
          @Override
          public int numDocs() {
            return batchNumDocs;
          }

          @Override
          public Iterable<Column> columns() {
            return columns;
          }
        });
    writer.commit();
  }

  /**
   * Returns one pair's columns: the doc-map column resolving from-side doc ids to to-side doc
   * ids, and the edges companion columns. The edges columns are written even when the pair
   * maps nothing, so a once-built pair is detectable in the join index and never rebuilt.
   */
  private static List<Column> createJoinColumns(JoinColumnModel mapping, String pairFieldName) {
    return List.of(
        ordMapBatch(pairFieldName, mapping),
        edgesColumn(AIJoinUtil.FROM_EDGES_PREFIX + pairFieldName, mapping.edges().fromDocEdges()),
        edgesColumn(AIJoinUtil.TO_EDGES_PREFIX + pairFieldName, mapping.edges().toDocEdges()),
        edgesColumn(
            AIJoinUtil.TO_COUNT_PREFIX + pairFieldName, new int[] {mapping.edges().toCount()}));
  }

  private static LongColumn edgesColumn(String fromEdgesFieldName, int[] fromDocEdges) {
    return new LongColumn(
        fromEdgesFieldName, TO_DOCS_FIELD_TYPE, Column.Density.SPARSE, NumericKind.INT) {
      @Override
      public LongTupleCursor tuples() {
        return new LongTupleCursor() {
          int i = -1;

          @Override
          public int nextDoc() {
            if (++i < fromDocEdges.length) {
              return 0; // try to put both vals at the doc 0
            }
            return DocIdSetIterator.NO_MORE_DOCS;
          }

          @Override
          public long longValue() {
            return (long) fromDocEdges[i];
          }
        };
      }
    };
  }

  /**
   * A column persisting a doc mapping: batch-local doc number is the from-side doc id and the
   * SORTED_NUMERIC docvalue is the matching to-side doc id. From docs without a match keep -1
   * in the array and get no value, hence the column is sparse.
   */
  private static Column ordMapBatch(String fieldName, JoinColumnModel mapping) {
    return new LongColumn(
        AIJoinUtil.TO_DOC_VAL_BY_FROM_DOCNUM + fieldName,
        TO_DOCS_FIELD_TYPE,
        Column.Density.SPARSE,
        NumericKind.INT) {
      @Override
      public LongTupleCursor tuples() {
        // a fresh cursor each call, per the Column contract, wrapping a fresh DV instance too
        SortedNumericDocValues values = mapping.toDocByFromDoc();
        return new LongTupleCursor() {
          @Override
          public int nextDoc() {
            try {
              return values.nextDoc();
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          }

          @Override
          public long longValue() {
            try {
              return values.nextValue();
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          }
        };
      }
    };
  }
}
