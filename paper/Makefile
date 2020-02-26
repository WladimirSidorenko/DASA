##################################################################
# Variables
BIBFILE := bibliography.bib
MAIN_FILE := dasa.pdf

##################################################################
# Special Targets
.DELETE_ON_ERROR:

.PHONY: all clean

##################################################################
# Main Target
all: ${MAIN_FILE}

clean:
	-rm -f *.lot *.log *.aux *.out *.bbl *.lof *.blg

##################################################################
# Main Paper File
${MAIN_FILE}: %.pdf: %.tex ${BIBFILE} $(wildcard *.tex)
	@set -e; \
	pdflatex -halt-on-error -output-directory ${@D} $< && \
	bibtex -min-crossrefs=100 $(basename $<) && \
	pdflatex -halt-on-error -output-directory ${@D} $< && pdflatex -halt-on-error \
	-output-directory ${@D} $<
