import csv

def format_ner_data(data_path):
    """
    Reformat NER data as
    text = [
        [w_1, w_2, ..., w_k1],
        [w_1, w_2, ..., w_k2],
        ...
        [w_1, w_2, ..., w_kn],
    ]
    labels = [
        [ent_1, ent_2, ..., ent_k1],
        [ent_1, ent_2, ..., ent_k2],
        ...
        [ent_1, ent_2, ..., ent_kn]
    ]
    """

    with open(data_path, "r") as f:
        x = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        text_arr = []
        label_arr = []

        curr_sent = []
        curr_label = []
        for row in x:
            if len(row) > 0:
#                 print(row[0])
                curr_sent.append(row[0])
#                 print(row[1])
                curr_label.append(row[1])
            else:
                text_arr.append(curr_sent)
                label_arr.append(curr_label)
                curr_sent = []
                curr_label = []

    # Not foul-proof
    #     test_label = [
    #     ["a", "a", "c"],
    #     ["b", "c", "d"]
    # ]
    # Will return {'a': 1, 'c': 1, 'b': 0, 'd': 2}
    label_dict = {label: i for label_sent in label_arr for i, label in enumerate(label_sent)}

    return text_arr, label_arr, label_dict