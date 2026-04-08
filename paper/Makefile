MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all quick clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex appendices/*.tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

quick:
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,fls,fdb_latexmk,synctex.gz}
	rm -f sections/*.aux appendices/*.aux
