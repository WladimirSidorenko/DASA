Files
-----

* PotTS - files of the [PotTS sentiment
  corpus](https://github.com/WladimirSidorenko/PotTS);

* SB10k - files of the [Cieliebak's Sentiment
  Corpus](http://aclweb.org/anthology/W17-1106).

All files were taken from the [CGSA
Project](https://github.com/WladimirSidorenko/CGSA) as state
`https://github.com/WladimirSidorenko/CGSA` and converted to JSON
using the script `DASA/scripts/tsv2json`, e.g.:

```shell
./scripts/tsv2json data/SB10k/dev/*.tsv
```

where `*.tsv` were copied over from `CGSA`.
