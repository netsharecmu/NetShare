import argparse

def main():
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a list of syn_df folders from DIFFERENT configs on the SAME dataset
    parser.add_argument("--syn_df_folders", nargs="+", type=str)
    parser.add_argument('--num_chunks', type=int)

    args = parser.parse_args()
    main(args)