#!/usr/bin/env python
__author__ = "Paul Duckworth"

import rospy
import os, sys
import argparse
import numpy as np
import getpass, datetime
import cPickle as pickle
import pyLDAvis
import human_activities.onlineldavb as onlineldavb
import human_activities.utils as utils


def load_required_files(path):
    """load all the required data"""

    data_path = os.path.join(path, date)
    lda_path = os.path.join(data_path, "oLDA")

    print "path load: %s " % lda_path
    code_book, graphlets, feature_space = utils.load_learning_files_all(data_path)

    with open(lda_path + "/olda.p", 'r') as f:
        olda = pickle.load(f)
    gamma = np.loadtxt(lda_path + '/gamma.dat')
    return olda, gamma, code_book, graphlets, feature_space

def create_codebook_images(codebook, im_path=None, dbg=False):
    for cnt, (g_name, g) in enumerate(codebook.items()):
        if dbg: print "\n", hash , g
        if cnt % 1000 is 0: print cnt
        # if not os.path.isfile(os.path.join(im_path, g_name+".png")):
        graphlet2dot(g, g_name, im_path)

def get_dic_codebook(date, code_book, graphlets, create_graphlet_images=False):
    dictionary_codebook = {}
    for hash, graph in zip(code_book, graphlets):
        g_name = "{:20.0f}".format(hash).lstrip()
        dictionary_codebook[g_name] = graph

    if create_graphlet_images:
        #investigate_features(dictionary_codebook)
        image_path = '/home/' + getpass.getuser() + '/SkeletonDataset/Learning/accumulate_data/' + date + '/oLDA/images'
        create_codebook_images(dictionary_codebook, image_path)
    return dictionary_codebook

def vis_topic_model(n_top_words, olda, gamma, X, graphlets, html_file):
    print dir(olda)

    for i in dir(olda):
        print olda.i

    # print "\n>", olda._lambda
    # print "\n>", gamma
    # print type(graphlets), len(graphlets.keys())

    data = {'topic_term_dists': olda._lambda,
            'doc_topic_dists': gamma,
            'doc_lengths': len(graphlets.keys()),
            'vocab': graphlets.keys(),
            'term_frequency': X}

    feature_freq = (X != 0).sum(axis=0)
    doc_lengths = (X != 0).sum(axis=1)

    print "phi: %s. theta: %s. nd: %s. vocab: %s. Mw: %s" \
        %( olda._lambda.shape, gamma.shape, doc_lengths.shape, len(graphlets.keys()), len(feature_freq))

    # topic_freq       = np.dot(model.doc_topic_.T, doc_lengths)
    # print "<", topic_freq
    #
    # print model.doc_topic_.T.shape
    # print (model.doc_topic_.T * doc_lengths).shape
    # print ((model.doc_topic_.T * doc_lengths).T).shape, type((model.doc_topic_.T * doc_lengths).T)
    # print (model.doc_topic_.T * doc_lengths).T.sum()
    #
    # topic_freq = (model.doc_topic_.T * doc_lengths).T.sum()
    # print "<<: ", topic_freq
    # a = topic_freq / topic_freq.sum()
    # print "<", a

    # topic_proportion = (a).sort_values(ascending=False)
    # print ">>:", topic_proportion
    vis_data = pyLDAvis.prepare(olda._lambda, gamma, doc_lengths, graphlets.keys(), feature_freq)
    print "PyLDAVis ran. output: %s" % html_file
    # print ">>", vis_data
    pyLDAvis.save_html(vis_data, html_file)

    """INVESTIGATE TOPICS"""
    topic_word = olda._lambda
    print("\ntype(topic_word): {}".format(type(topic_word)))
    print("shape: {}".format(topic_word.shape))
    topics = {}
    for i, topic_dist in enumerate(topic_word):
        objs = []
        topic_words = np.array(graphlets.keys())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        #print('Topic {}: {}'.format(i, ' '.join( [repr(i) for i in topic_words] )))
        for j in [graphlets[k] for k in topic_words]:
            objs.extend(object_nodes(j)[0])
        topics[i] = objs
        print('Topic {}: {}'.format(i, list(set(objs))))

    """INVESTIGATE DOCUMENTS"""
    # doc_topic = gamma
    # print("\ntype(doc_topic): {}".format(type(doc_topic)))
    # print("shape: {}".format(doc_topic.shape))
    # for n in xrange(0):
    #     sum_pr = sum(doc_topic[n,:])
    #     doc_main_topics_ = (doc_topic[n,:]>0.2)
    #     probable_topics = []
    #     for cnt, i in enumerate(doc_main_topics_):
    #         if i:
    #             probable_topics.append(cnt)
    #     print "%s = %s. %s " % (n, true_labels[n], probable_topics)
    #     # print "n = %s. %s " % (n, (doc_topic[n,:]>0.3))
    #     # print("document: {} sum: {}".format(n, sum_pr))
    return


def object_nodes(graph):
    object_nodes = []
    num_of_eps = 0
    for node in graph.vs():
        if node['node_type'] == 'object':
            if node['name'] not in ["hand", "torso"]:
                object_nodes.append(node['name'])
        if node['node_type'] == 'spatial_relation':
            num_of_eps+=1

    return object_nodes, num_of_eps

def graphlet2dot(graph, g_name, path):
    """Modified version of the Graph2dot function for Activity Graphs"""

    # Write the graph to dot file
    # Can generate a graph figure from this .dot file using the 'dot' command
    # dot -Tpng input.dot -o output.png

    out_dot_file = os.path.join(path, g_name + ".dot")

    dot_file = open(out_dot_file, 'w')
    dot_file.write('digraph activity_graph {\n')
    dot_file.write('    size = "40,40";\n')
    dot_file.write('    node [fontsize = "16", shape = "box", style="filled", fillcolor="aquamarine"];\n')
    dot_file.write('    ranksep=5;\n')
    # Create temporal nodes
    dot_file.write('    subgraph _1 {\n')
    dot_file.write('    rank="source";\n')


    ##Because it's not an Activity Graph - need to create all these things:
    temporal_nodes=[]
    temporal_ids=[]
    for node in graph.vs():
        if node['node_type'] == 'temporal_relation':
            temporal_nodes.append(node)
            temporal_ids.append(node.index)
            #print node, node.index

    spatial_nodes = []
    spatial_ids = []
    for node in graph.vs():
        if node['node_type'] == 'spatial_relation':
            spatial_nodes.append(node)
            spatial_ids.append(node.index)
            #print node, node.index

    object_nodes = []
    object_ids = []
    for node in graph.vs():
        if node['node_type'] == 'object':
            object_nodes.append(node)
            object_ids.append(node.index)
            #print node, node.index

    temp_spatial_edges = []
    spatial_obj_edges = []

    for edge in graph.es():
        if edge.source in object_ids and edge.target in spatial_ids:
            spatial_obj_edges.append((edge.source, edge.target))
        elif edge.source in spatial_ids and edge.target in object_ids:
            spatial_obj_edges.append((edge.source, edge.target))
        elif edge.source in temporal_ids and edge.target in spatial_ids:
            temp_spatial_edges.append((edge.source, edge.target))
        elif edge.source in spatial_ids and edge.target in temporal_ids:
            temp_spatial_edges.append((edge.source, edge.target))
        else:
            print "what's this?", edge.source, edge.target

    #Build Graph image

    for tnode in temporal_nodes:
        dot_file.write('    %s [fillcolor="white", label="%s", shape=ellipse];\n' %(tnode.index, tnode['name']))

    dot_file.write('}\n')

    # Create spatial nodes
    dot_file.write('    subgraph _2 {\n')
    dot_file.write('    rank="same";\n')
    for rnode in spatial_nodes:
        dot_file.write('    %s [fillcolor="lightblue", label="%s"];\n' %(rnode.index, rnode['name']))
    dot_file.write('}\n')

    # Create object nodes
    dot_file.write('    subgraph _3 {\n')
    dot_file.write('    rank="sink";\n')
    for onode in object_nodes:
        dot_file.write('%s [fillcolor="tan1", label="%s"];\n' %(onode.index, onode['name']))
    dot_file.write('}\n')

    # Create temporal to spatial edges
    for t_edge in temp_spatial_edges:
        #print t_edge[0],t_edge[1]
        dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(t_edge[0], t_edge[1]))

    # Create spatial to object edges
    for r_edge in spatial_obj_edges:
        dot_file.write('%s -> %s [arrowhead = "normal", color="red"];\n' %(r_edge[0], r_edge[1]))
    dot_file.write('}\n')
    dot_file.close()

    # creat a .png then remove the .dot to save memory. Fix the png to either 900 or 1500, then whitespace it to fix the size
    foofile = os.path.join(path, "foo.png")
    outfile = os.path.join(path, g_name + ".png")
    os.system("dot -Tpng -Gsize=9,15\! -Gdpi=100 %s -o %s " % (out_dot_file,foofile) )
    os.system("convert %s -gravity center -background white -extent 900x1500 %s" % (foofile, outfile))
    os.system("rm %s" % out_dot_file)
    # os.system("rm %s" % foofile)

if __name__ == "__main__":

    rospy.init_node("offline_activity_investigation")

    parser = argparse.ArgumentParser(description='Offline Human Activity Investigation')
    parser.add_argument('date', type=str, help='date of learned topic model')
    parser.add_argument('top_words', type=int, help='number of top words to consider')
    args = parser.parse_args()
    date = args.date
    top_words = args.top_words
    rospy.loginfo("Offline Human Activity Investigation: %s" % date)

    path = '/home/' + getpass.getuser() + '/SkeletonDataset/Learning/accumulate_data'
    olda, gamma, code_book, graphlets, feature_space = load_required_files(path)
    print "lamda:", olda._lambda.shape
    print "gamma:", gamma.shape

    print code_book

    sys.exit(1)
    dict_code_book = get_dic_codebook(date, code_book, graphlets, False)
    out_file = os.path.join(path, date, "oLDA", "ldavis_topic_model.html")
    vis_topic_model(top_words, olda, gamma, feature_space, dict_code_book, out_file)
