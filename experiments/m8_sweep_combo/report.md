# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.2360  0.0193       0.2579  0.0145
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.1978  0.0249       0.2477  0.0091
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1890  0.0291       0.2383  0.0124
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.2284  0.0179       0.2537  0.0140
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2172  0.0220       0.2483  0.0084
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2337  0.0216       0.2564  0.0152
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2117  0.0208       0.2491  0.0060
```
