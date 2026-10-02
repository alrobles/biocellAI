# M1 — cell-only vs text-grounded (held-out donors)

Dataset: `pbmc3k` | seeds: [np.int64(0)]

## Results

```
                                macro_f1     balanced_acc    
                                    mean std         mean std
arm                  label_mode                              
cell_only_supervised NaN          0.8885 NaN       0.8708 NaN
grounded_probe       False        0.2486 NaN       0.2459 NaN
                     True         0.8217 NaN       0.7767 NaN
grounded_zeroshot    False        0.0914 NaN       0.1253 NaN
                     True         0.6890 NaN       0.8231 NaN
```

## Reading

- `grounded_zeroshot label_mode=True` = captions name the cell type
  (leakage probe — upper bound, NOT a grounding result).
- `label_mode=False` = marker-only captions (the honest grounding arm).
- `grounded_probe` = frozen encoder + linear probe on train donors.
- `cell_only_supervised` = same encoder, end-to-end supervised.
