Files
-----

* PotTS - files of the
  [PotTS](https://github.com/WladimirSidorenko/PotTS) sentiment
  corpus;

* SB10k - files of the
  [Cieliebak's](http://aclweb.org/anthology/W17-1106) sentiment
  corpus.

All files were taken from the [CGSA
Project](https://github.com/WladimirSidorenko/CGSA) at state
`https://github.com/WladimirSidorenko/CGSA` and converted to JSON
using the script `DASA/scripts/tsv2json`, e.g.:

```shell
./scripts/tsv2json data/SB10k/dev/*.tsv
```

where `*.tsv` filed had been previously copied over from `CGSA`.

Discourse segmentation was obtained using the discourse segmenter
package `dsegmenter` from mate parse terrs.  The corresponding
segments are stored in the files `data/{PotTS,SB10k}/raw/*.seg` and
were merged into the main json files using the script
`scripts/add_segmentation`, e.g.:

```shell
./scripts/add_segmentation data/SB10k/test/test.json data/SB10k/raw/corpus_v1.0.preprocessed.seg

for f in data/PotTS/*/*.json ; do \
  ./scripts/add_segmentation $f data/PotTS/raw/$(basename ${f%json})seg > 1 \
  && mv 1 $f; \
done
```
