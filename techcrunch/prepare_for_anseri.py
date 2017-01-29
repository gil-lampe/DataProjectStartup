# coding=utf-8
"""
Turn downloaded data into anseri dataset

Requires Python 3.5

:author: Andrew B Godbehere
:date: 5/30/16
"""

import anseri as ai
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory", type=str, help="Directory of content to import")
    parser.add_argument("dataset_name", type=str, help="Name for new dataset")

    subparsers = parser.add_subparsers(help="Specify the type of document splitting", dest='split_type')
    sent_parser = subparsers.add_parser("sentence")

    window_parser = subparsers.add_parser("window")
    window_parser.add_argument("--num_words", required=True, type=int)
    window_parser.add_argument("--step", type=int)

    doc_parser = subparsers.add_parser("document")

    args = parser.parse_args()
    if args.split_type is None:
        args.split_type = "sentence"

    ai.enable_progress()
    aljazeera = ai.Dataset("aljazeera")

    if args.split_type == 'sentence':
        splitter = ai.dataset.SentenceSplitter('content')
    elif args.split_type == 'window':
        if args.step is None:
            args.step = args.num_words

        splitter = ai.dataset.MovingWindowSplitter('content', args.num_words, args.step)
    else:
        splitter = ai.dataset.DocumentSplitter('content')

    new_dataset = ai.Dataset.from_iterable(ai.data.data_streams.from_directory(args.input_directory),
                                           args.dataset_name, ["content"], "created",
                                           splitter=splitter,
                                           preprocessor=aljazeera.preprocessor)

    print("Imported {} sentences into dataset named {}.".format(new_dataset.num_documents, new_dataset.name))
