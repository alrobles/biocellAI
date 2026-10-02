# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(10), np.int64(11), np.int64(12)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.2002  0.0493       0.2508  0.0102
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.1970  0.0522       0.2460  0.0138
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2053  0.0615       0.2513  0.0187
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.2013  0.0547       0.2460  0.0195
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2127  0.0621       0.2531  0.0229
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2015  0.0559       0.2461  0.0201
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.2125  0.0636       0.2536  0.0233
```
