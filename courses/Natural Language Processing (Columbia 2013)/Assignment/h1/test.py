import collections
def modify_input_find_rare_words(input_train, modified_train, modify=False):
    out = file(modified_train, "w")
    input_lines = file(input_train,"r").read().splitlines()
    words = [str.split(line, ' ')[0] for line in input_lines if line]
    counter = collections.Counter(words)
    rare_words = [w for w in counter if counter[w] < 5]
    #print rare_words
    print len(input_lines)
    if modify:
		out_str = ''
		for line in input_lines:
			if line:
				w, tag = str.split(line, ' ')
				out_str += ('_RARE_' if w in rare_words else w) + ' ' + tag + '\n'
			else:
				out_str += '\n'
		out.write(out_str)
		out.close()    
    return rare_words

def compute_MLE_params_and_predict_tags(rare_words, input_count, input_test, output):
    e, q = {}, {}
    for line in file(input_count,"r").read().splitlines():
        words = str.split(line, ' ')
        if words[1] == 'WORDTAG':
            e[words[3], words[2]] = int(words[0])
        elif words[1] == '1-GRAM':
            q[words[2]] = int(words[0])
        elif words[1] == '2-GRAM':
            q[words[2], words[3]] = int(words[0])
        elif words[1] == '3-GRAM':
            q[words[2], words[3], words[4]] = int(words[0])
    for (x,y) in e:
        e[x,y] = e[x,y] / float(q[y])
    # print e
	del_keys = []
    for ngram in q.keys():
		n = len(ngram)
		if n == 3:
			y1, y2, y3 = ngram
			q[y1, y2, y3] = q[y1, y2, y3] / float(q[y1, y2])
		else:
			del_keys.append(ngram) 
    for ngram in del_keys:
		del q[ngram]
    # print q
    
    tags = {}
    for (x,y) in e:
        _, prob = tags.get(x, (None, 0))
        if prob <= e[x,y]:
            tags[x] = (y, e[x,y])            
    test, out = file(input_test,"r"), file(output, "w")
    for x in test.read().splitlines():
        if x:
            key = '_RARE_' if (x in rare_words) or (not x in tags) else x
            out.write(x + ' ' + str(tags[key][0]) + '\n')
        else:
            out.write('\n')
    out.close()    
    
    return (e, q, rare_words, tags)

def run_baseline():	
	path = './'    
	rare_words = modify_input_find_rare_words(path + 'gene.train', path + 'gene.mod.train')
	e, q, rare_words, tags = compute_MLE_params_and_predict_tags(rare_words, path+'gene.counts', path+'gene.dev', path+'gene_dev.p1.out')
	# python eval_gene_tagger.py gene.key gene_dev.p1.out

from math import log, exp

def log1(x):
    return log(x) if x > 0 else -float('inf')

def K(tags, k):
    return tags if k > 0 else ['*']

from count_freqs import sentence_iterator, simple_conll_corpus_iterator
def viterbi_decoding(x, q, e):
    D, bp = {}, {}
    n = len(x)
    y = [0]*n
    tags = set(sum([list(k) for k in q.keys()],[])) - set(['*', 'STOP']) 
    #print tags
    D[0, '*', '*'] = 1
    for k in range(1,n+1):
        for u in K(tags,k-1):
            for v in K(tags,k):
                #print x[k-1], v
                #D[k,u,v], bp[k,u,v] = max([(D[k-1,w,u]*q[w,u,v]*e[x[k-1],v], w) for w in K(tags, k-2)])
                #if k <= 2:
                #    for w in K(tags, k-2): print k, w, u, v, x[k-1],q[w,u,v],D[k-1,w,u],e.get((x[k-1],v),0),exp(log1(D[k-1,w,u])+log1(q[w,u,v])+log1(e.get((x[k-1],v),0)))
                #D[k,u,v], bp[k,u,v] = max([(D[k-1,w,u]*q[w,u,v]*e.get((x[k-1],v),0), w) for w in K(tags, k-2)])
                D[k,u,v], bp[k,u,v] = max([(exp(log1(D[k-1,w,u])+log1(q[w,u,v])+log1(e.get((x[k-1],v),0))), w) for w in K(tags, k-2)])
    #_, y[n-2], y[n-1] = max([(D[n,u,v]*q[u,v,'STOP'],u,v) for u in K(tags, n-1) for v in K(tags,n)])
    _, y[n-2], y[n-1] = max([(exp(log1(D[n,u,v])+log1(q[u,v,'STOP'])),u,v) for u in K(tags, n-1) for v in K(tags,n)])
    #print D
    for k in range(n-2, 0, -1):
        y[k-1] = bp[k+2, y[k], y[k+1]]
    return y

def run_viterbi():	

	n = 3
	path = './'	
	rare_words = modify_input_find_rare_words(path + 'gene.train', path + 'gene.mod.train')
	e, q, rare_words, tags = compute_MLE_params_and_predict_tags(rare_words, path+'gene.counts', path+'gene.dev', path+'gene_dev.p1.out')
	train, test, out = file(path+'gene.train',"r"), file(path+'gene.dev',"r"), file(path+'gene_dev.p2.out', "w")
	count = 0
	for sentence in sentence_iterator(simple_conll_corpus_iterator(test)):
		x = [w[1] for w in sentence]
		#print x
		x_new = ['_RARE_' if (x[i] in rare_words) or (not x[i] in tags) else x[i] for i in range(len(x))] 
		y = viterbi_decoding(x_new, q, e)
		#print len(x), len(y), y
		for i in range(len(x)):
			out.write(x[i] + ' ' + y[i] + '\n')
		out.write('\n')
		count += 1
	out.close() 
	print 'dev sentence count', count

	count = 0
	for sentence in sentence_iterator(simple_conll_corpus_iterator(train)):
		count += 1
	print 'train sentence count', count

from count_freqs import Hmm	
def compute_MLE_params_with_rare_groups(input_train, input_test):
    # Initialize a trigram counter
    counter = Hmm(3)
    # Collect counts
    counter.train(file(input_train,"r"))  
    Counts = {}
    for (x,y) in counter.emission_counts:
        Counts[x] = Counts.get(x,0) + counter.emission_counts[(x,y)]
    #print Counts    
    rare_words = [x for x in Counts if Counts[x] < 5]
    for (x,y) in counter.emission_counts.keys():
        if x in rare_words:
            type = '_NUM_' if any([ch.isdigit() for ch in x]) else '_RARE_'
            if type == '_RARE_': type = '_ACAP_' if all([ch.isupper() for ch in x]) else '_RARE_'
            if type == '_RARE_': type = '_LCAP_' if x[-1].isupper() else '_RARE_'
            counter.emission_counts[type,y] = counter.emission_counts.get((type,y), 0) + counter.emission_counts[x,y]
            del counter.emission_counts[x,y]
    e = {(x,y):counter.emission_counts[(x,y)] / float(counter.ngram_counts[0][(y,)]) for (x,y) in counter.emission_counts}
    # print e
    q = {(y1,y2,y3):counter.ngram_counts[2][y1,y2,y3]/float(counter.ngram_counts[1][y1,y2]) for (y1,y2,y3) in counter.ngram_counts[2]}
    # print q
    tags = {}
    for (x,y) in e:
        _, prob = tags.get(x, (None, 0))
        if prob <= e[x,y]:
            tags[x] = (y, e[x,y])      
    return (e, q, rare_words, tags)


def build_source(x, y):
	src = 'digraph genealogy { \n \
    size = "50,20"; \n \
    node [fontsize = "16", shape = "circle", style="filled", fillcolor="steelblue"]; \n \
	\n \
    subgraph _1 { \n \
     rank="same"; \n \
    '
	xids, yids = ['x' + str(i) for i in range(len(x))], ['y' + str(i) for i in range(len(y))]
	for i in range(len(x)):
		src += xids[i] + ' [ fillcolor="lightblue", label=< <B>' + x[i] + '</B> >];\n'
	for i in range(len(x)-1):
		src += xids[i] + ' -> ' + xids[i+1] + ' [dir="right"];\n'
	src += '} \n \
	\n \
    subgraph _2 {  \n \
    rank="same"; \n \
	'
	for i in range(len(y)):
		src += yids[i] + ' [ fillcolor="lightpink", label=< <B>' + y[i] + '</B> > ];\n'
	src += '} \n \
	\n \
	'
	for i in range(len(x)):
		src += xids[i] + ' -> ' + yids[i] + ' [dir="onormal", color="darkgreen"];\n'
	
	src += '\n}'
	return src
	
from graphviz import Source	
def save_hmm_graph(x, y, n):
	src = Source(build_source(x, y), format='png')#'jpg')
	src.render('test-output/hmm' + str(n), view=False)
	
def run_viterbi2():    
	path = './'    
	e, q, rare_words, tags = compute_MLE_params_with_rare_groups(path+'gene.train', path+'gene.dev')

	n = 3
	test, out = file(path+'gene.dev',"r"), file(path+'gene_dev.p3.out', "w")
	sc = 1
	for sentence in sentence_iterator(simple_conll_corpus_iterator(test)):
		x = [w[1] for w in sentence]
		#print x
		x_new = [w for w in x]
		for i in range(len(x_new)):
			if (x_new[i] in rare_words) or (not x_new[i] in tags):
				type = '_NUM_' if any([ch.isdigit() for ch in x_new[i]]) else '_RARE_'
				if type == '_RARE_': type = '_ACAP_' if all([ch.isupper() for ch in x_new[i]]) else '_RARE_'
				if type == '_RARE_': type = '_LCAP_' if x_new[i][-1].isupper() else '_RARE_'
				x_new[i] = type            
		y = viterbi_decoding(x_new, q, e)
		#print len(x), len(y), y
		for i in range(len(x)):
			out.write(x[i] + ' ' + y[i] + '\n')
		print sc	
		save_hmm_graph(x, y, sc)
		sc += 1
		out.write('\n')
	out.close() 

run_viterbi2()

def plot_accuracy():
	import pandas as pd
	df = pd.DataFrame({'Tagger':'Baseline', 'Precision':0.158861,'Recall':0.660436, 'F1-Score':0.256116}, index=[0])
	df = df.append(pd.DataFrame({'Tagger':'Viterbi', 'Precision':0.541555,'Recall':0.314642, 'F1-Score':0.398030}, index=[1]))
	df = df.append(pd.DataFrame({'Tagger':'Viterbi with Rareword groups', 'Precision':0.534940,'Recall':0.345794, 'F1-Score':0.420057}, index=[2]))
	df = df[['Tagger', 'Precision', 'Recall', 'F1-Score']]
	df.head()

	df = pd.melt(df, id_vars=['Tagger'], value_vars=['Precision', 'Recall', 'F1-Score'])
	df.head(10)
	import seaborn as sns
	sns.barplot(x="variable", y="value", hue="Tagger", data=df)
	sns.plt.show()

#convert -resize 4800x450 -extent 4800x450 -delay 100 -loop 0 *.png sol.gif		
#convert -resize 4800x450^ -delay 100 -loop 0 *.png sol.gif

### graphviz
# dot -Tpng test.dot > test.png
#from graphviz import Source
#src = Source('digraph "the holy hand grenade" { rankdir=LR; 1 -> 2 -> 3 -> lob }', format='png')
#src.render('test-output/holy-grenade', view=False)
