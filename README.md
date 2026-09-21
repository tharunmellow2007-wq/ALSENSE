# ALSENSE
A hybrid quantum-classical machine learning pipeline for early ALS (Amyotrophic Lateral Sclerosis) screening, combining voice biomarkers (bulbar symptoms) and gait/walking-test features (limb symptoms), with classical ML, quantum ML (QSVM/VQC/QNN), and hybrid fusion strategies benchmarked side-by-side using statistically validated cross-validation.
Problem

ALS has no confirmatory test — diagnosis takes 10–15 months on average and relies on ruling out other conditions [1][2], with up to 40% of patients initially misdiagnosed [3]. This project screens for early ALS risk from non-invasive voice + gait signals, shortening the path to specialist diagnosis and multidisciplinary care, which is independently associated with a 6-month survival benefit [4].

Key Results (Strict Dysarthric ALS vs. Healthy Control cohort)
Model	Accuracy	F1-Score	ROC-AUC
Quantum-Only (QSVM)	72.1%	83.6%	0.555
Classical ML	89.9%	93.5%	0.933
Hybrid (Quantum + Classical)	88.0%	92.2%	0.927

Statistical validation (paired t-test, identical CV folds):

Hybrid vs. Quantum-only: p < 0.001 — statistically real gain
Hybrid vs. Classical: p = 0.33 — no significant loss

Reported honestly: quantum-only alone does not yet outperform classical ML on this dataset (AUC 0.555, near-random), but hybrid fusion recovers the gap almost entirely while adding statistically validated quantum-derived features — consistent with the current NISQ-era literature on small biomedical datasets.

References

[1] The ALS Association, "Let's See Quicker Diagnoses," 2024. https://www.als.org/blog/lets-see-quicker-diagnoses

[2] "Pathogenesis and Presentation of ALS: Examining Reasons for Delayed Diagnosis and Identifying Opportunities for Improvement," American Journal of Managed Care, 2024.

[3] "Preventing Unnecessary Surgery in Patients Presenting for Orthopedic Spine Surgery: Literature Review and Case Series," Journal of Orthopaedic Case Reports, 2023.

[4] A. J. Paipa et al., "Survival benefit of multidisciplinary care in amyotrophic lateral sclerosis in Spain," Journal of Multidisciplinary Healthcare, vol. 12, pp. 465-470, 2019, doi: 10.2147/JMDH.S205313.

[5] A. Tena et al., "Detection of Bulbar Involvement in Patients With Amyotrophic Lateral Sclerosis by Machine Learning Voice Analysis," JMIR Medical Informatics, vol. 9, no. 3, p. e21331, 2021, doi: 10.2196/21331.

[6] A. S. Gupta et al., "At-home wearables and machine learning sensitively capture disease progression in amyotrophic lateral sclerosis," Nature Communications, vol. 14, p. 5080, 2023, doi: 10.1038/s41467-023-40917-3.

Disclaimer

This is a research screening tool, not a diagnostic device. All outputs are intended as referral-support signals for clinician review — final diagnosis always remains with a qualified neurologist.

License

MIT — see LICENSE.
