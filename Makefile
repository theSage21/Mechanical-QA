getalldatasets: getsquaddata
getsquaddata:
	test -d data || mkdir data
	wget -O data/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
	wget -O data/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
.PHONY: getalldatasets getsquaddata
