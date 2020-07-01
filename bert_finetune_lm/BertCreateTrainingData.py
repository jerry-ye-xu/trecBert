import pandas as pd

# def write_training_data(file_path, out_path):
#     df = pd.read_pickle(file_path)
#     i = 0
#     with open(out_path, "w") as f:
#         for r in df.iterrows():
#             if i > 10:
#                 break
#             i +=1
#             print(r[1]["brief_title"])
#             print(r[1]["brief_summary"])
#             f.write(r[1]["brief_title"] + ".\n")
#             f.write(r[1]["brief_summary"] + "\n")
#             # print("\n\n")
#             # f.write("\n\n")
#         # [f.write(x) for x in df["brief_title"] + " " + df["brief_summary"] + "\n\n"]

def write_training_data(file_path, out_path):
    df = pd.read_pickle(file_path)
    df = df.drop_duplicates(["brief_title"])
    i = 0
    with open(out_path, "w") as f:
        for r in df.iterrows():
            if i > 500:
                break
            i += 1
            print(r[1]["brief_title"])
            print(r[1]["brief_summary"])
            f.write(str(r[1]["brief_title"]) + ".\n")
            f.write(str(r[1]["brief_summary"]) + "\n")

if __name__ == "__main__":
    file_path = "./data/trials_topics_combined_all_years.pickle"
    out_path = "./data/trials_raw_sample.txt"
    write_training_data(file_path, out_path)

    with open("./data/test.txt", "w") as f:
        for i in range(10):
            f.write("hello\n")
            f.write("world\n")