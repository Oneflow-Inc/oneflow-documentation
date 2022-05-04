from collections import OrderedDict
import argparse
from extract_code_block import *


def run_block_item(block_dict: OrderedDict, file_path=None):
    exec_error_msg = """    markdown file: {0}
    codeblock index: {1}
    Code:{2}"""
    for index in block_dict:
        try:
            exec(block_dict[index], globals(), globals())
        except:
            print("    ****EXEC ERROR****")
            print(
                exec_error_msg.format(
                    file_path, index, bytes(block_dict[index], encoding="utf-8")
                )
            )
            print("")
            raise RuntimeError("markdown test fails")


def run_markdown_codes(file_path, index):
    codes = get_all_python_blocks(file_path)
    picked_codes = pickup_blocks(codes, index)
    run_block_item(picked_codes, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run python code in markdown files")
    parser.add_argument(
        "--markdown_file", type=str, help="the path of markdown file. eg: ./sample.md"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="all",
        help='the index set of code blocks. eg: [0, 1, 2]. Default for "all"',
    )

    args = parser.parse_args()
    file_path = args.markdown_file
    if args.index != "all":
        index = eval(args.index)
    else:
        index = args.index
    run_markdown_codes(file_path, index)
