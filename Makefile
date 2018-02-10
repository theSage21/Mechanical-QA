prereq: datasets glove
glove: datafolder
	test -e data/glove.6B.zip || http://nlp.stanford.edu/data/glove.6B.zip
	test -e data/glove.6B.50d.txt || unzip -p data/glove.6B.zip glove.6B.50d.txt > data/glove.6B.50d.txt
datasets: squad
datafolder:
	test -d data || mkdir data
squad: datafolder
	test -e data/train-v1.1.json || wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	test -e data/dev-v1.1.json || wget -O data/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
	test -e data/evalsquad.py  || wget -O data/evalsquad.py https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/
.PHONY: prereq glove datasets datafolder squad
