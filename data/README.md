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
