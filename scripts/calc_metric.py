import argparse

from sklearn.metrics import classification_report
import pandas as pd


def main(fileOrigin, fileResult):
    y_test = pd.read_csv(fileOrigin, sep="\t")["label"].values
    with open(fileResult, "r") as fr:
        y_pred = []
        for line in fr:
            line = line.split("\t")
            if len(line) != 2 or line[0].isalpha() or line[1].isalpha():
                continue
            y_pred.append(int(line[1]))
    rp = classification_report(y_test, y_pred)
    print("report:\n{}".format(rp))


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser(
        description="calculate classfication report",
        formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument("-origin", help="The path to the original text file.", required=True)
    parser.add_argument("-result", help="The path to the result text file.", required=True)
    args = parser.parse_args()
    # Run the program.
    main(args.origin, args.result)
