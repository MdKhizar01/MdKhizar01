# Reconstruction validation

Verified in October 2026 with Python 3, OpenJDK 17 source-file mode, NumPy, pandas, scikit-learn, Pillow, and Flask.

| Project | Checks completed |
| --- | --- |
| DDoS local pipeline | Generated 1000 synthetic records, trained on 750, evaluated on 250, exported metrics/predictions/models |
| Diabetes demo | Synthetic training, home-page rendering, invalid-input rejection, prediction API, SQLite history |
| Medicinal leaf baseline | Generated 72 synthetic images across three classes, feature extraction, stratified training/evaluation, artifact export |
| Distributed gradient | 1000 items and eight agents, equality run, inequality feasibility, projection bounds, progress toward centralized reference |
| Java IR | Indexed three sample documents including a TREC DOCNO, persisted/reloaded index, ranked robot.txt first for `graph search robot` |
| Sorting | Sizes 0, 1, 100, 1000; random, sorted, reverse-sorted inputs; all four outputs matched Python sorted |
| Litecho | Home-page rendering, TXT extraction, unsupported-file rejection, empty-text rejection, MP3 response with an injected fake speech backend |
| Urban dashboard | Inline JavaScript syntax checked with Node; browser interaction/rendering not tested |

All Python files passed syntax parsing. Diabetes and Litecho inline browser scripts passed JavaScript syntax checks. Model scores on synthetic demo data are not real-world performance evidence.

Not verified: live Spark/Cassandra/AWS infrastructure, real dataset performance, gTTS network synthesis, Tesseract OCR, real PDF extraction through Litecho, or browser interaction/accessibility audits. Optional dependencies and deployment setup require separate validation.
