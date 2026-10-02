# M2 — ablation: grounding source x text encoder (held-out donors)

Dataset: `/beegfs/a474r867/bioai/data/seaad_mtg_v1c_240026_s0.h5ad` | seeds: [np.int64(0), np.int64(1), np.int64(2)]
label_mode=False throughout (honest grounding only).

## Results

```
                                                                                     macro_f1         balanced_acc        
                                                                                         mean     std         mean     std
arm                       caption_mode text_model                                                                         
cell_only_supervised      -            -                                               0.1950  0.0287       0.3054  0.0704
grounded_probe            type_llm     all-MiniLM-L6-v2                                0.1979  0.0300       0.3061  0.0722
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1944  0.0268       0.3018  0.0745
grounded_zeroshot         type_llm     all-MiniLM-L6-v2                                0.1994  0.0333       0.3041  0.0636
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1938  0.0309       0.2941  0.0649
grounded_zeroshot_matched type_llm     all-MiniLM-L6-v2                                0.2003  0.0324       0.3059  0.0644
                                       cambridgeltl/SapBERT-from-PubMedBERT-fulltext   0.1946  0.0308       0.2949  0.0643
```
