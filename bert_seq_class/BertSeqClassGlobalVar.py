SUBSET_COLUMNS = [
    "id", "year",
    "gene", "disease",
    "brief_summary", "brief_title",
    "topic", "label"
]

global_var = {
    "subset_columns": SUBSET_COLUMNS,
    "year": "year",
    "label": "label",
    "raw_topic": "topic", # Used for process_baseline_raw
    "topic": "topic",
    "id_topic": "id_topic", 
    "attrib_seq": "ty_id",
    "doc_id": "id",
    "seq_a_baseline": "d_and_g",
    "seq_b_baseline": "brief_t_and_s",
    "seq_a_expansion": "qe_all"
}

global_var_cv = {
    "subset_columns": SUBSET_COLUMNS,
    "year": "year",
    "label": "label",
    "raw_topic": "topic", 
    "topic": "topics_all", 
    "id_topic": "id_topics_all",
    # "topics_all": "topics_all",
    "attrib_seq": "ty_id",
    "doc_id": "id",
    "seq_a_baseline": "d_and_g",
    "seq_b_baseline": "brief_t_and_s",
    "seq_a_expansion": "qe_all",
    "seq_all_sent": "seq_all_sent",
    "orig2yrtop_dict_path": "./data/cv_files/orig2yrtop_dict.pickle"
}
