# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_allregions_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.1849  0.0467       0.2839  0.0446
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.1898  0.0411       0.2955  0.0603
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1908  0.0371       0.3015  0.0691
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1927  0.0411       0.2909  0.0568
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1922  0.0395       0.2932  0.0605
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.1941  0.0400       0.2933  0.0577
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1930  0.0391       0.2948  0.0603
```
