| run | questions | recall@5 | mrr | qa_em | qa_f1 | rougeL | bertscore_f1 | retrieval_seconds | query_seconds | peak_rss_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 30 | 0.300 | 0.357 | 0.067 | 0.103 | 0.014 | 0.642 | 0.763 | 1.803 | 948.035 |
| bge | 30 | 0.435 | 0.462 | 0.100 | 0.158 | not measured | not measured | 0.878 | 2.316 | 948.789 |
| bm25 | 30 | 0.239 | 0.306 | 0.100 | 0.107 | not measured | not measured | 0.774 | 1.601 | 1016.809 |
| chunk-large | 30 | 0.301 | 0.333 | 0.100 | 0.136 | not measured | not measured | 0.792 | 1.853 | 1010.559 |
| chunk-small | 30 | 0.221 | 0.306 | 0.033 | 0.070 | not measured | not measured | 1.001 | 2.050 | 989.387 |
| dense | 30 | 0.281 | 0.271 | 0.067 | 0.104 | not measured | not measured | 0.819 | 1.938 | 1000.785 |
| distilbart | 30 | 0.300 | 0.357 | 0.067 | 0.103 | 0.138 | 0.774 | 0.777 | 1.876 | 1929.383 |
| flan-base | 30 | 0.300 | 0.357 | 0.067 | 0.103 | 0.079 | 0.682 | 0.853 | 2.008 | 1599.008 |
| reranked | 30 | 0.317 | 0.332 | 0.067 | 0.088 | not measured | not measured | 2.769 | 3.799 | 1015.844 |
