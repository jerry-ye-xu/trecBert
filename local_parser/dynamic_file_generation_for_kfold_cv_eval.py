import pandas as pd
import xml.dom.minidom as md

from BertSeqClassGlobalVar import global_var_cv

TOPICS_TO_ALWAYS_REMOVE = [10, 112, 113]

def reduce_topics_for_eval(
    topics_path,
    save_tmp_topics_path,
    training_topic_set, tag):
    DOMTree = md.parse(topics_path)
    topics = DOMTree.documentElement.getElementsByTagName("topic")

    # Need to check that index corresponds to topic number
    # otherwise there will be erroneous behaviour.
    for i in range(len(topics)):
        assert int(topics[i].getAttribute("number")) == (i + 1)
    
    # Topic 10 of 2017, 32 and 33 of 2019 are not part of qrels assessment. 
    # They will always be removed
    for topic_num in (training_topic_set + TOPICS_TO_ALWAYS_REMOVE):
        print(f"Removing topic number: {topic_num}")
        print(f"Checking topic XML attribute: {topics[topic_num - 1].getAttribute(tag)}")
        topics[topic_num - 1].parentNode.removeChild(topics[topic_num - 1])

    with open(f"{save_tmp_topics_path}", "w") as f:
        DOMTree.writexml(f)

def reduce_qrels_for_eval(
    qrels_path, save_tmp_qrels_path,
    col_names, training_topic_set):
    qrels_all = pd.read_csv(
        qrels_path,
        names=col_names, sep=" "
    )
    qrels_test = qrels_all[~qrels_all[global_var_cv["topic"]].isin(training_topic_set + TOPICS_TO_ALWAYS_REMOVE)]
    qrels_test.to_csv(
        f"{save_tmp_qrels_path}",
        header=False, index=False,
        sep=" "
    )
